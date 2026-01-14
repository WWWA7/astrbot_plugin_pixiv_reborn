import asyncio
import aiohttp
import aiofiles
import base64
import subprocess
import zipfile
import tempfile
import io
import random
from pathlib import Path
from typing import Any, Optional, List, Tuple
from PIL import Image as PILImage

from astrbot.api import logger
from astrbot.api.message_components import Image, Plain, Node, Nodes
from pixivpy3 import AppPixivAPI

from .config import PixivConfig
from .tag import filter_illusts_with_reason, FilterConfig
from .config import smart_clean_temp_dir, clean_temp_dir


_config = None
_temp_dir = None

def init_pixiv_utils(client: AppPixivAPI, config: PixivConfig, temp_dir: Path):
    """初始化 PixivUtils 模块的全局变量"""
    global _config, _temp_dir
    _config = config
    _temp_dir = temp_dir


def filter_items(items, tag_label, excluded_tags=None):
    """统一过滤插画/小说的辅助方法"""
    config = FilterConfig(
        r18_mode=_config.r18_mode,
        ai_filter_mode=_config.ai_filter_mode,
        display_tag_str=tag_label,
        return_count=_config.return_count,
        logger=logger,
        show_filter_result=_config.show_filter_result,
        excluded_tags=excluded_tags or []
    )
    
    return filter_illusts_with_reason(items, config)


def generate_safe_filename(title: str, default_name: str = "pixiv") -> str:
    """生成安全的文件名，移除特殊字符"""
    safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).rstrip()
    return safe_title if safe_title else default_name


def obfuscate_image_data(img_bytes: bytes) -> bytes:
    """破坏图片哈希值"""
    try:
        img = PILImage.open(io.BytesIO(img_bytes))
        
        width, height = img.size
        if width > 1 and height > 1:
            pixel = list(img.getpixel((0, 0)))
            if isinstance(pixel, (list, tuple)) and len(pixel) >= 3:
                change = random.choice([-1, 1])
                new_val = max(0, min(255, pixel[0] + change))
                new_pixel = list(pixel)
                new_pixel[0] = new_val
                img.putpixel((0, 0), tuple(new_pixel))
            elif isinstance(pixel, int):
                change = random.choice([-1, 1])
                img.putpixel((0, 0), max(0, min(255, pixel + change)))

        output_buffer = io.BytesIO()
        fmt = img.format if img.format else "JPEG"
        
        if fmt.upper() in ["JPEG", "JPG"]:
            img.save(output_buffer, format=fmt, quality=random.randint(98, 100))
        else:
            img.save(output_buffer, format=fmt)
            
        return output_buffer.getvalue()
        
    except Exception as e:
        logger.warning(f"Pixiv 插件：破坏图片哈希失败 - {e}")
        return img_bytes


def get_illust_page_urls(illust, max_pages: int = 0) -> Tuple[List[str], int, int]:
    """
    获取作品的所有页面URL
    
    Args:
        illust: 作品对象
        max_pages: 最大页数限制，0表示不限制
    
    Returns:
        (url列表, 实际发送页数, 总页数)
    """
    total_pages = illust.page_count
    
    if total_pages == 1:
        # 单页作品
        original_url = None
        if hasattr(illust, 'meta_single_page') and illust.meta_single_page:
            original_url = getattr(illust.meta_single_page, 'original_image_url', None)
        if not original_url:
            original_url = getattr(illust.image_urls, _config.image_quality, None)
            if not original_url:
                original_url = getattr(illust.image_urls, 'large', None)
        return [original_url] if original_url else [], 1, 1
    
    # 多页作品
    urls = []
    pages_to_send = total_pages if max_pages == 0 else min(total_pages, max_pages)
    
    for i in range(pages_to_send):
        if i < len(illust.meta_pages):
            page = illust.meta_pages[i]
            url = getattr(page.image_urls, _config.image_quality, None)
            if not url:
                url = getattr(page.image_urls, 'large', None)
            if url:
                urls.append(url)
    
    return urls, pages_to_send, total_pages


def build_page_hint(sent_pages: int, total_pages: int) -> str:
    """构建页数提示信息"""
    if total_pages > 1 and sent_pages < total_pages:
        return f"\n[本作品共 {total_pages} 页，已发送前 {sent_pages} 页]"
    elif total_pages > 1:
        return f"\n[本作品共 {total_pages} 页]"
    return ""


def build_ugoira_info_message(illust, metadata, gif_info, detail_message: str = None) -> str:
    """构建动图信息消息"""
    ugoira_info = "🎬 动图作品\n"
    ugoira_info += f"标题: {illust.title}\n"
    ugoira_info += f"作者: {illust.user.name}\n"
    ugoira_info += f"帧数: {len(metadata.frames)}\n"
    ugoira_info += f"GIF大小: {gif_info.get('size', 0) / 1024 / 1024:.2f} MB\n"
    
    if detail_message:
        lines = detail_message.split('\n')
        for line in lines:
            if line.startswith('标签:'):
                ugoira_info += f"{line}\n"
                break
    
    ugoira_info += f"作品链接: https://www.pixiv.net/artworks/{illust.id}\n\n"
    return ugoira_info


async def download_image(session: aiohttp.ClientSession, url: str, headers: dict = None) -> Optional[bytes]:
    """下载图片数据（增强版）"""
    default_headers = {"Referer": "https://app-api.pixiv.net/"}
    if headers:
        default_headers.update(headers)

    if "i.pximg.net" not in url:
        try:
            async with session.get(url, headers=default_headers, proxy=_config.proxy or None) as response:
                if response.status == 200:
                    return await response.read()
                return None
        except Exception as e:
            logger.error(f"Pixiv 插件：非官方图片下载失败 - {e}")
            return None

    sources = [
        ("i.pximg.net", True),
        ("i.pixiv.re", False),
        ("i.pixivel.moe", False),
    ]

    for domain, use_proxy in sources:
        current_url = url.replace("i.pximg.net", domain)
        current_proxy = _config.proxy if (use_proxy and _config.proxy) else None

        log_prefix = "官方源" if domain == "i.pximg.net" else f"反代源({domain})"
        logger.debug(f"Pixiv 插件：尝试下载图片 [{log_prefix}]...")

        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with session.get(current_url, headers=default_headers, proxy=current_proxy, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.read()
                    if data:
                        logger.info(f"Pixiv 插件：图片下载成功 [{log_prefix}]")
                        return data
                else:
                    logger.warning(f"Pixiv 插件：下载失败 [{log_prefix}] 状态码: {response.status}")
        
        except asyncio.TimeoutError:
            logger.warning(f"Pixiv 插件：下载超时 [{log_prefix}]")
        except Exception as e:
            logger.warning(f"Pixiv 插件：下载异常 [{log_prefix}] - {e}")
    
    logger.error(f"Pixiv 插件：所有源均下载失败，放弃下载。URL: {url}")
    return None


async def download_illust_all_pages(session: aiohttp.ClientSession, illust, max_pages: int = 0) -> Tuple[List[bytes], int, int]:
    """
    下载作品的所有页面图片
    
    Returns:
        (图片数据列表, 实际发送页数, 总页数)
    """
    urls, sent_pages, total_pages = get_illust_page_urls(illust, max_pages)
    
    images_data = []
    for url in urls:
        if url:
            img_data = await download_image(session, url)
            if img_data:
                # 如果是原图质量，破坏哈希
                if _config.image_quality == "original":
                    img_data = await asyncio.to_thread(obfuscate_image_data, img_data)
                images_data.append(img_data)
    
    return images_data, sent_pages, total_pages


async def process_ugoira_for_content(client: AppPixivAPI, session: aiohttp.ClientSession,
                                   illust, detail_message: str = None) -> Optional[dict]:
    """处理动图并返回内容字典"""
    try:
        ugoira_metadata = await asyncio.to_thread(client.ugoira_metadata, illust.id)
        if not ugoira_metadata or not hasattr(ugoira_metadata, 'ugoira_metadata'):
            return None
        
        metadata = ugoira_metadata.ugoira_metadata
        if not hasattr(metadata, 'zip_urls') or not metadata.zip_urls.medium:
            return None
        
        zip_url = metadata.zip_urls.medium
        
        zip_data = await download_image(session, zip_url)
        if not zip_data:
            return None
        
        safe_title = generate_safe_filename(illust.title, "ugoira")
        gif_result = await _convert_ugoira_to_gif(zip_data, metadata, safe_title, illust.id)
        
        if gif_result:
            gif_data, gif_info = gif_result
            try:
                ugoira_info = build_ugoira_info_message(illust, metadata, gif_info, detail_message)
                return {
                    'gif_data': gif_data,
                    'ugoira_info': ugoira_info
                }
            except Exception as e:
                logger.error(f"Pixiv 插件：处理动图GIF时发生错误 - {e}")
                return None
        else:
            return None
            
    except Exception as e:
        logger.error(f"Pixiv 插件：处理动图时发生错误 - {e}")
        return None


async def authenticate(client: AppPixivAPI) -> bool:
    """尝试使用配置的凭据进行 Pixiv API 认证"""
    try:
        if _config.refresh_token:
            await asyncio.to_thread(client.auth, refresh_token=_config.refresh_token)
            return True
        else:
            logger.error("Pixiv 插件：未提供有效的 Refresh Token，无法进行认证。")
            return False
    except Exception as e:
        logger.error(f"Pixiv 插件：认证/刷新时发生错误 - {e}")
        return False


async def send_pixiv_image(
    client: AppPixivAPI,
    event: Any,
    illust,
    detail_message: str = None,
    show_details: bool = True,
    send_all_pages: bool = False,
):
    """通用Pixiv图片下载与发送函数，支持多页作品"""
    if hasattr(illust, 'type') and illust.type == 'ugoira':
        logger.info(f"Pixiv 插件：检测到动图作品 - ID: {illust.id}")
        async for result in send_ugoira(client, event, illust, detail_message):
            yield result
        return
    
    await smart_clean_temp_dir(_temp_dir, probability=0.1, max_files=20)

    max_pages = _config.max_pages_per_illust if _config.max_pages_per_illust > 0 else 0
    
    # 如果是 send_all_pages 模式或多页作品，发送多张图
    if send_all_pages or illust.page_count > 1:
        async with aiohttp.ClientSession() as session:
            images_data, sent_pages, total_pages = await download_illust_all_pages(session, illust, max_pages)
            
            if not images_data:
                yield event.plain_result(f"图片下载失败（所有源均不可用），仅发送信息：\n{detail_message or ''}")
                return
            
            # 构建页数提示
            page_hint = build_page_hint(sent_pages, total_pages)
            final_message = (detail_message or "") + page_hint
            
            # 构建消息：多张图片 + 详情
            image_components = [Image.fromBytes(img_data) for img_data in images_data]
            
            if show_details and final_message:
                image_components.append(Plain(final_message))
            
            yield event.chain_result(image_components)
    else:
        # 单页作品，保持原有逻辑
        async with aiohttp.ClientSession() as session:
            images_data, sent_pages, total_pages = await download_illust_all_pages(session, illust, 1)
            
            if not images_data:
                yield event.plain_result(f"图片下载失败（所有源均不可用），仅发送信息：\n{detail_message or ''}")
                return
            
            if show_details and detail_message:
                yield event.chain_result([Image.fromBytes(images_data[0]), Plain(detail_message)])
            else:
                yield event.chain_result([Image.fromBytes(images_data[0])])


async def send_ugoira(client: AppPixivAPI, event: Any, illust, detail_message: str = None):
    """处理动图（ugoira）的下载和发送"""
    await smart_clean_temp_dir(_temp_dir, probability=0.1, max_files=20)
    
    try:
        async with aiohttp.ClientSession() as session:
            content = await process_ugoira_for_content(client, session, illust, detail_message)
            
            if content:
                gif_data = content['gif_data']
                ugoira_info = content['ugoira_info']
                
                yield event.chain_result([Image.fromBytes(gif_data), Plain(ugoira_info)])
                
                if _config.is_fromfilesystem and event.get_platform_name() == "aiocqhttp" and event.get_group_id():
                    try:
                        from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
                        if isinstance(event, AiocqhttpMessageEvent):
                            client_bot = event.bot
                            group_id = event.get_group_id()
                            safe_title = generate_safe_filename(illust.title, "ugoira")
                            file_name = f"{safe_title}_{illust.id}.gif"
                            gif_base64 = base64.b64encode(gif_data).decode('utf-8')
                            base64_uri = f"base64://{gif_base64}"
                            await client_bot.upload_group_file(group_id=group_id, file=base64_uri, name=file_name)
                    except Exception as e:
                        logger.error(f"Pixiv 插件：上传群文件失败 - {e}")
            else:
                yield event.plain_result("动图处理失败")

    except Exception as e:
        logger.error(f"Pixiv 插件：处理动图时发生错误 - {e}")
        yield event.plain_result(f"处理动图时发生错误: {str(e)}")


async def _convert_ugoira_to_gif(zip_data, metadata, safe_title, illust_id):
    """将动图ZIP文件转换为GIF格式"""
    temp_dir = None
    try:
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, timeout=10)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("Pixiv 插件：ffmpeg不可用")
            return None
        
        temp_dir = tempfile.mkdtemp(prefix=f"pixiv_ugoira_{illust_id}_", dir=_temp_dir)
        zip_path = Path(temp_dir) / f"{safe_title}_{illust_id}.zip"
        async with aiofiles.open(zip_path, "wb") as f:
            await f.write(zip_data)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        if not hasattr(metadata, 'frames') or not metadata.frames:
            return None
        
        frames_dir = Path(temp_dir)
        frame_files = []
        
        for i, frame in enumerate(metadata.frames):
            possible_names = [f"frame_{i:06d}.jpg", f"frame_{i:06d}.png", f"{i:06d}.jpg", f"{i:06d}.png", f"frame_{i}.jpg", f"frame_{i}.png"]
            frame_file = None
            for name in possible_names:
                potential_file = frames_dir / name
                if potential_file.exists():
                    frame_file = potential_file
                    break
            
            if frame_file:
                duration = getattr(frame, 'delay', 100)
                frame_files.append(f"file '{frame_file}'\nduration {duration/1000}")
        
        if not frame_files:
            return None
        
        concat_file = Path(temp_dir) / "frames.txt"
        async with aiofiles.open(concat_file, "w", encoding='utf-8') as f:
            await f.write("\n".join(frame_files))
        
        output_gif = Path(temp_dir) / f"{safe_title}_{illust_id}.gif"
        
        cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', str(concat_file), '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-gifflags', '+transdiff', str(output_gif)]
        
        process = await asyncio.create_subprocess_exec(*cmd, cwd=str(temp_dir), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
        except asyncio.TimeoutError:
            process.kill()
            return None

        if process.returncode != 0 or not output_gif.exists():
            return None
        
        try:
            with open(output_gif, 'rb') as f:
                gif_data = f.read()
            return gif_data, {'frames': len(metadata.frames), 'size': len(gif_data)}
        except Exception:
            return None
            
    except Exception as e:
        logger.error(f"Pixiv 插件：转换动图异常 - {e}")
        return None


async def send_forward_message(client: AppPixivAPI, event, images, build_detail_message_func):
    """直接下载图片并组装 nodes，支持多页作品"""
    batch_size = 10
    nickname = "PixivBot"
    await clean_temp_dir(_temp_dir, max_files=20)
    
    max_pages = _config.max_pages_per_illust if _config.max_pages_per_illust > 0 else 0
    
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i : i + batch_size]
        nodes_list = []
        async with aiohttp.ClientSession() as session:
            for img in batch_imgs:
                if hasattr(img, 'type') and img.type == 'ugoira':
                    detail_message = build_detail_message_func(img) if _config.show_details else None
                    content = await process_ugoira_for_content(client, session, img, detail_message)
                    if content:
                        node_content = [Image.fromBytes(content['gif_data']), Plain(content['ugoira_info'])]
                    else:
                        node_content = [Plain("动图处理失败")]
                else:
                    detail_message = build_detail_message_func(img)
                    
                    # 下载多页图片
                    images_data, sent_pages, total_pages = await download_illust_all_pages(session, img, max_pages)
                    
                    node_content = []
                    
                    if images_data:
                        for img_data in images_data:
                            node_content.append(Image.fromBytes(img_data))
                        
                        # 添加页数提示
                        page_hint = build_page_hint(sent_pages, total_pages)
                        final_message = detail_message + page_hint
                        
                        if _config.show_details:
                            node_content.append(Plain(final_message))
                    else:
                        node_content.append(Plain(f"图片下载失败，仅发送信息\n{detail_message}"))
                   
                nodes_list.append(Node(name=nickname, content=node_content))
        if nodes_list:
            yield event.chain_result([Nodes(nodes=nodes_list)])

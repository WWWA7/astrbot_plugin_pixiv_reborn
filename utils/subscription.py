import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from astrbot.api import logger
from astrbot.core.message.message_event_result import MessageChain
from pixivpy3 import AppPixivAPI
from ..utils.pixiv_utils import (
    filter_items,
    download_image,
)

from .database import get_all_subscriptions, update_last_notified_id
from .tag import build_detail_message


class SubscriptionService:
    def __init__(self, client_wrapper, pixiv_config, context):
        self.client_wrapper = client_wrapper
        self.client = client_wrapper.client_api
        self.pixiv_config = pixiv_config
        self.context = context
        self.scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")
        self.job = None
        
        # 定义目标文件夹路径
        log_dir = r"C:\Users\Administrator\Desktop\AstrBot\pixiv"
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self.log_file = os.path.join(log_dir, "pixiv_subscription.log")
        
        # 创建专用的临时目录（避免短路径问题）
        self.temp_dir = Path(log_dir) / "temp_images"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _save_log(self, content: str):
        """写入本地订阅日志"""
        try:
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"[{time_str}] {content}\n"
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception as e:
            logger.error(f"Pixiv 订阅日志写入失败: {e}")

    def start(self):
        """启动后台任务"""
        if not self.scheduler.running:
            self._save_log("=== 订阅服务启动 ===")
            self._save_log(f"配置: image_quality={self.pixiv_config.image_quality}, "
                          f"size_limit_enabled={self.pixiv_config.image_size_limit_enabled}, "
                          f"size_limit_mb={self.pixiv_config.image_size_limit_mb}, "
                          f"platform_instance_name={self.pixiv_config.platform_instance_name}")
            self._save_log(f"临时目录: {self.temp_dir}")
            self.job = self.scheduler.add_job(
                self.check_subscriptions,
                "interval",
                minutes=self.pixiv_config.subscription_check_interval_minutes,
                next_run_time=datetime.now() + timedelta(seconds=10),
            )
            self.scheduler.start()

    def stop(self):
        """停止后台任务"""
        if self.scheduler.running:
            self._save_log("=== 订阅服务停止 ===")
            self.scheduler.shutdown()
            logger.info("订阅检查服务已停止。")

    async def check_subscriptions(self):
        """检查所有订阅并推送更新"""
        self._save_log("开始新一轮订阅检查...")
        
        if not await self.client_wrapper.authenticate():
            err_msg = "订阅检查失败：Pixiv API 认证失败。"
            logger.error(err_msg)
            self._save_log(err_msg)
            return

        subscriptions = get_all_subscriptions()
        if not subscriptions:
            self._save_log("当前无任何订阅记录。")
            return

        self._save_log(f"获取到 {len(subscriptions)} 条订阅记录，开始逐个检查。")

        for sub in subscriptions:
            try:
                if sub.sub_type == "artist":
                    await self.check_artist_updates(sub)
            except Exception as e:
                err_msg = f"检查订阅 {sub.sub_type}: {sub.target_id} 时发生错误: {e}"
                logger.error(err_msg)
                self._save_log(err_msg)
                import traceback
                self._save_log(traceback.format_exc())
            await asyncio.sleep(5)
        
        self._save_log("本轮订阅检查完成。")

    async def check_artist_updates(self, sub):
        """检查画师更新"""
        api: AppPixivAPI = self.client
        json_result = await asyncio.to_thread(api.user_illusts, sub.target_id)

        if not json_result or not json_result.illusts:
            return

        new_illusts = []
        for illust in json_result.illusts:
            if illust.id > sub.last_notified_illust_id:
                new_illusts.append(illust)
            else:
                break

        if new_illusts:
            count = len(new_illusts)
            new_ids = [str(i.id) for i in new_illusts]
            self._save_log(f"画师 [{sub.target_name}] 发现 {count} 张新作品! IDs: {', '.join(new_ids)}")
            
            new_illusts.reverse()
            latest_id = new_illusts[-1].id
            
            update_last_notified_id(sub.chat_id, sub.sub_type, sub.target_id, latest_id)
            self._save_log(f"更新数据库 last_id -> {latest_id}")

            for illust in new_illusts:
                filtered_illusts, reason = filter_items(
                    [illust], f"画师订阅: {sub.target_name}"
                )
                if filtered_illusts:
                    self._save_log(f"准备推送作品 PID: {illust.id} -> 会话: {sub.session_id}")
                    await self.send_update(sub, filtered_illusts[0])
                    await asyncio.sleep(2)
                else:
                    self._save_log(f"作品 PID: {illust.id} 被过滤，原因: {reason}")

    def _fix_session_id(self, session_id: str) -> str:
        """
        修复 session_id，将旧格式的平台类型转换为配置的平台实例名称
        例如：aiocqhttp:GroupMessage:123 -> 喵喵ll:GroupMessage:123
        """
        # 如果没有配置平台实例名称，直接返回原始值
        platform_name = self.pixiv_config.platform_instance_name
        if not platform_name:
            self._save_log(f"[WARN] 未配置 platform_instance_name，使用原始 session_id")
            return session_id
        
        try:
            parts = session_id.split(":")
            if len(parts) < 3:
                return session_id
            
            platform_identifier = parts[0]
            
            # 需要转换的旧格式平台类型列表
            old_platform_types = [
                "aiocqhttp", "onebot", "cqhttp", "gocqhttp", 
                "napcat", "llonebot", "shamrock"
            ]
            
            # 检查是否是旧格式（平台类型而非实例名称）
            if platform_identifier.lower() in old_platform_types:
                # 替换为配置的平台实例名称
                new_session_id = f"{platform_name}:{':'.join(parts[1:])}"
                self._save_log(f"[DEBUG] 转换 session_id: {session_id} -> {new_session_id}")
                return new_session_id
            
            return session_id
            
        except Exception as e:
            self._save_log(f"[ERROR] 修复 session_id 时出错: {e}")
            return session_id

    async def send_update(self, sub, illust):
        """发送更新通知"""
        import aiohttp
        
        tmp_path = None
        try:
            session_id_str = sub.session_id
            
            # 修复：使用配置的平台实例名称转换 session_id
            session_id_str = self._fix_session_id(session_id_str)
            self._save_log(f"[DEBUG] PID {illust.id}: 使用 session_id: {session_id_str}")
            
            # 构建详情消息
            detail_message = (
                f"您订阅的画师 [{sub.target_name}] 有新作品啦！\n"
            )
            detail_message += build_detail_message(illust, is_novel=False)

            # 获取图片URL对象
            if illust.page_count > 1 and illust.meta_pages:
                url_obj = illust.meta_pages[0].image_urls
            else:
                url_obj = illust.image_urls
            
            # 按质量优先级获取URL并下载
            quality_preference = ["original", "large", "medium"]
            start_index = (
                quality_preference.index(self.pixiv_config.image_quality)
                if self.pixiv_config.image_quality in quality_preference
                else 0
            )
            
            img_data = None
            size_limit_mb = self.pixiv_config.image_size_limit_mb
            size_limit_enabled = self.pixiv_config.image_size_limit_enabled
            
            self._save_log(f"[DEBUG] PID {illust.id}: 开始下载图片, page_count={illust.page_count}")
            
            async with aiohttp.ClientSession() as session:
                for quality in quality_preference[start_index:]:
                    image_url = None
                    
                    # 修复：正确获取原图URL
                    if quality == 'original':
                        # 对于单页作品，原图URL在 meta_single_page 中
                        if hasattr(illust, 'meta_single_page') and illust.meta_single_page:
                            image_url = getattr(illust.meta_single_page, 'original_image_url', None)
                        # 对于多页作品，原图URL在 url_obj.original 中
                        if not image_url and hasattr(url_obj, 'original'):
                            image_url = getattr(url_obj, 'original', None)
                    else:
                        # large 和 medium 直接从 url_obj 获取
                        if hasattr(url_obj, quality):
                            image_url = getattr(url_obj, quality, None)

                    if not image_url:
                        self._save_log(f"[DEBUG] PID {illust.id}: 质量 {quality} 无URL，跳过")
                        continue

                    self._save_log(f"[DEBUG] PID {illust.id}: 尝试下载质量 {quality}")
                    downloaded_data = await download_image(session, image_url)
                    
                    if downloaded_data:
                        size_mb = len(downloaded_data) / (1024 * 1024)
                        self._save_log(f"[DEBUG] PID {illust.id}: 质量 {quality} 下载成功，大小 {size_mb:.2f}MB")
                        
                        if size_limit_enabled and size_mb > size_limit_mb and quality != "medium":
                            self._save_log(f"图片 PID {illust.id} 大小 {size_mb:.2f}MB 超限，尝试降级")
                            continue
                        
                        img_data = downloaded_data
                        self._save_log(f"图片 PID {illust.id} 最终使用质量 {quality}，大小 {size_mb:.2f}MB")
                        break
                    else:
                        self._save_log(f"[DEBUG] PID {illust.id}: 质量 {quality} 下载失败")

            if not img_data:
                self._save_log(f"[ERROR] PID {illust.id}: 所有质量下载失败")
                return

            # 保存到临时文件
            tmp_path = str(self.temp_dir / f"pixiv_{illust.id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
            with open(tmp_path, 'wb') as f:
                f.write(img_data)
            
            file_size = os.path.getsize(tmp_path)
            self._save_log(f"[DEBUG] PID {illust.id}: 文件已保存到 {tmp_path}，大小 {file_size} 字节")
            
            # 构建消息链
            if self.pixiv_config.show_details:
                message_chain = MessageChain().file_image(tmp_path).message(detail_message)
            else:
                message_chain = MessageChain().file_image(tmp_path)
            
            # 发送消息
            result = await self.context.send_message(session_id_str, message_chain)
            
            if result:
                self._save_log(f"推送成功: PID {illust.id}")
            else:
                self._save_log(f"[ERROR] PID {illust.id}: send_message 返回 False，请检查 platform_instance_name 配置")

        except Exception as e:
            err_msg = f"发送订阅更新时出错 (PID {illust.id}): {e}"
            logger.error(err_msg)
            self._save_log(err_msg)
            import traceback
            logger.error(traceback.format_exc())
            self._save_log(traceback.format_exc())
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    await asyncio.sleep(3)
                    os.remove(tmp_path)
                    self._save_log(f"[DEBUG] PID {illust.id}: 临时文件已清理")
                except Exception as e:
                    self._save_log(f"[DEBUG] 清理临时文件失败: {e}")

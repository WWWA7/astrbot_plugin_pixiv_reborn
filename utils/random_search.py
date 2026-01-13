import asyncio
import random
import os
import tempfile
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from astrbot.api import logger
from astrbot.core.message.message_event_result import MessageChain

from .database import (
    get_all_random_search_groups,
    get_random_tags,
    filter_sent_illusts,
    add_sent_illust,
    cleanup_old_sent_illusts,
    get_schedule_time,
    set_schedule_time,
    remove_schedule_time,
    get_all_schedule_times,
    get_all_random_ranking_groups,
    get_random_rankings,
)
from .tag import (
    build_detail_message,
    FilterConfig,
    validate_and_process_tags,
    filter_illusts_with_reason,
    sample_illusts,
)
from .pixiv_utils import download_image
import aiohttp


class RandomSearchService:
    def __init__(self, client_wrapper, pixiv_config, context):
        self.client_wrapper = client_wrapper
        self.client = client_wrapper.client_api
        self.pixiv_config = pixiv_config
        self.context = context

        self.scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")
        self.job = None
        self.execution_locks = {}
        self.global_execution_lock = asyncio.Lock()
        self.task_queue = asyncio.Queue()
        self.is_queue_processor_running = False

    def start(self):
        """启动后台任务"""
        if not self.scheduler.running:
            self.job = self.scheduler.add_job(
                self._scheduler_tick,
                "interval",
                minutes=1,
                next_run_time=datetime.now() + timedelta(seconds=10),
            )
            self.scheduler.add_job(
                self._cleanup_task, "cron", hour=2, minute=0
            )

            self.scheduler.start()
            logger.info("Pixiv 随机搜索服务已启动。")
            logger.info(f"配置: image_quality={self.pixiv_config.image_quality}, "
                       f"size_limit_enabled={self.pixiv_config.image_size_limit_enabled}, "
                       f"size_limit_mb={self.pixiv_config.image_size_limit_mb}")
            self._load_existing_schedules()

    def _load_existing_schedules(self):
        """从数据库加载现有的调度时间"""
        try:
            schedules = get_all_schedule_times()
            logger.info(f"从数据库加载了 {len(schedules)} 个群组的调度时间")
        except Exception as e:
            logger.error(f"加载调度时间失败: {e}")

    def stop(self):
        """停止后台任务"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            self.is_queue_processor_running = False
            logger.info("Pixiv 随机搜索服务已停止。")

    async def _scheduler_tick(self):
        """检查是否有群组需要执行搜索，并将其加入队列。"""
        if not self.client:
            return

        try:
            if not self.is_queue_processor_running:
                asyncio.create_task(self._task_queue_processor())
                self.is_queue_processor_running = True
                logger.info("RandomSearchService 队列处理器已启动")

            tag_groups = get_all_random_search_groups()
            ranking_groups = get_all_random_ranking_groups()
            groups = list(set(tag_groups + ranking_groups))

            now = datetime.now()
            pending_groups = []

            for chat_id in groups:
                if chat_id not in self.execution_locks:
                    self.execution_locks[chat_id] = False

                next_execution_time = get_schedule_time(chat_id)

                if next_execution_time is None:
                    min_interval = self.pixiv_config.random_search_min_interval
                    max_interval = self.pixiv_config.random_search_max_interval
                    if max_interval < min_interval:
                        max_interval = min_interval

                    delay_minutes = random.randint(min_interval, max_interval)
                    next_execution_time = now + timedelta(minutes=delay_minutes)
                    set_schedule_time(chat_id, next_execution_time)
                    logger.info(f"群组 {chat_id}: 首次调度随机搜索，将在 {delay_minutes} 分钟后执行")
                    continue

                if now >= next_execution_time and not self.execution_locks[chat_id]:
                    pending_groups.append(chat_id)

            for chat_id in pending_groups:
                try:
                    await self.task_queue.put(chat_id)
                    logger.info(f"群组 {chat_id}: 已加入随机搜索队列")
                except Exception as e:
                    logger.error(f"将群组 {chat_id} 加入队列失败: {e}")

        except Exception as e:
            logger.error(f"RandomSearchService 调度器 tick 出错: {e}")

    async def _task_queue_processor(self):
        """任务队列处理器，按顺序执行队列中的搜索任务。"""
        logger.info("RandomSearchService 任务队列处理器开始运行")

        while True:
            try:
                chat_id = await self.task_queue.get()

                async with self.global_execution_lock:
                    if self.execution_locks.get(chat_id, False):
                        logger.warning(f"群组 {chat_id} 已在执行状态，跳过本次任务")
                        self.task_queue.task_done()
                        continue

                    self.execution_locks[chat_id] = True

                    try:
                        logger.info(f"开始执行群组 {chat_id} 的随机搜索")
                        await self.execute_search_for_group(chat_id)

                        now = datetime.now()
                        min_interval = self.pixiv_config.random_search_min_interval
                        max_interval = self.pixiv_config.random_search_max_interval
                        if max_interval < min_interval:
                            max_interval = min_interval

                        next_interval = random.randint(min_interval, max_interval)
                        new_execution_time = now + timedelta(minutes=next_interval)
                        set_schedule_time(chat_id, new_execution_time)
                        logger.info(f"群组 {chat_id}: 随机搜索已执行。下次运行在 {next_interval} 分钟后。")

                    except Exception as e:
                        logger.error(f"执行群组 {chat_id} 的随机搜索时出错: {e}")
                    finally:
                        self.execution_locks[chat_id] = False
                        self.task_queue.task_done()

            except asyncio.CancelledError:
                logger.info("RandomSearchService 任务队列处理器被取消")
                break
            except Exception as e:
                logger.error(f"RandomSearchService 任务队列处理器出错: {e}")
                await asyncio.sleep(5)

    async def _cleanup_task(self):
        """定期清理过期记录的任务"""
        try:
            logger.info("开始清理过期的已发送作品记录...")
            days = self.pixiv_config.random_sent_illust_retention_days
            await asyncio.to_thread(cleanup_old_sent_illusts, days=days)
            logger.info("清理过期记录任务完成。")
        except Exception as e:
            logger.error(f"清理过期记录任务出错: {e}")

    async def execute_search_for_group(self, chat_id: str):
        """为特定群组执行随机搜索（标签或排行榜）"""
        tags = get_random_tags(chat_id)
        rankings = get_random_rankings(chat_id)

        if not tags and not rankings:
            return

        all_options = []
        for tag in tags:
            all_options.append(("tag", tag))
        for ranking in rankings:
            all_options.append(("ranking", ranking))

        selected = random.choice(all_options)

        if selected[0] == "tag":
            await self._execute_tag_search(chat_id, selected[1])
        else:
            await self._execute_ranking_search(chat_id, selected[1])

    async def _execute_tag_search(self, chat_id: str, selected_tag_entry):
        """执行标签搜索"""
        raw_tag = selected_tag_entry.tag
        session_id = selected_tag_entry.session_id

        logger.info(f"正在为群组 {chat_id} 执行随机标签搜索，标签: {raw_tag}")

        if not await self.client_wrapper.authenticate():
            logger.error(f"群组 {chat_id} 的随机搜索失败: 认证失败。")
            return

        tag_result = validate_and_process_tags(raw_tag)
        if not tag_result["success"]:
            logger.warning(f"标签 {raw_tag} 的随机搜索验证失败: {tag_result['error_message']}")
            return

        search_tags = tag_result["search_tags"]
        exclude_tags = tag_result["exclude_tags"]
        display_tags = tag_result["display_tags"]

        try:
            search_params = {
                "word": search_tags,
                "search_target": "partial_match_for_tags",
                "sort": "popular_desc",
                "filter": "for_ios",
                "req_auth": True,
            }

            all_illusts = []
            page_count = 0
            deep_search_depth = self.pixiv_config.deep_search_depth
            next_params = search_params.copy()

            while next_params:
                if deep_search_depth > 0 and page_count >= deep_search_depth:
                    break

                json_result = await asyncio.to_thread(
                    self.client.search_illust, **next_params
                )

                if not json_result or not hasattr(json_result, "illusts"):
                    break

                current_illusts = json_result.illusts
                if current_illusts:
                    all_illusts.extend(current_illusts)
                    page_count += 1
                    logger.info(
                        f"标签 {raw_tag} 的随机搜索：已获取第 {page_count} 页，找到 {len(current_illusts)} 个插画"
                    )
                else:
                    break

                next_url = json_result.next_url
                next_params = self.client.parse_qs(next_url) if next_url else None

                if next_params:
                    await asyncio.sleep(0.5)

            if not all_illusts:
                logger.info(f"标签 {raw_tag} 的随机搜索未返回结果。")
                return

            initial_count = len(all_illusts)
            logger.info(
                f"标签 {raw_tag} 的随机搜索完成，共获取 {page_count} 页，找到 {initial_count} 个插画，开始过滤处理..."
            )

            initial_illusts = filter_sent_illusts(all_illusts, chat_id)

            if not initial_illusts:
                logger.info(f"标签 {raw_tag} 的随机搜索过滤后无可用作品。")
                return

            config = FilterConfig(
                r18_mode=self.pixiv_config.r18_mode,
                ai_filter_mode=self.pixiv_config.ai_filter_mode,
                display_tag_str=f"随机:{display_tags}",
                return_count=self.pixiv_config.return_count,
                logger=logger,
                show_filter_result=self.pixiv_config.show_filter_result,
                excluded_tags=exclude_tags or [],
            )

            filtered_illusts, _ = filter_illusts_with_reason(initial_illusts, config)

            if not filtered_illusts:
                logger.info(f"标签 {raw_tag} 的随机搜索过滤后无符合条件的作品。")
                return

            illusts_to_send = sample_illusts(
                filtered_illusts, self.pixiv_config.return_count, shuffle=True
            )

            for illust in illusts_to_send:
                await self._send_illust(session_id, illust, chat_id)
                await asyncio.sleep(2)

        except Exception as e:
            logger.error(f"为群组 {chat_id} 执行随机标签搜索时出错: {e}")

    async def _execute_ranking_search(self, chat_id: str, ranking_config):
        """执行排行榜搜索"""
        mode = ranking_config.mode
        date = ranking_config.date
        session_id = ranking_config.session_id

        logger.info(
            f"正在为群组 {chat_id} 执行随机排行榜搜索，模式: {mode}, 日期: {date}"
        )

        if not await self.client_wrapper.authenticate():
            logger.error(f"群组 {chat_id} 的随机排行榜搜索失败: 认证失败。")
            return

        try:
            ranking_result = await asyncio.to_thread(
                self.client.illust_ranking, mode=mode, date=date
            )
            initial_illusts = ranking_result.illusts if ranking_result.illusts else []

            if not initial_illusts:
                logger.info(f"排行榜 {mode} 的随机搜索未返回结果。")
                return

            initial_illusts = filter_sent_illusts(initial_illusts, chat_id)

            if not initial_illusts:
                logger.info(f"排行榜 {mode} 的随机搜索过滤后无可用作品。")
                return

            config = FilterConfig(
                r18_mode=self.pixiv_config.r18_mode,
                ai_filter_mode=self.pixiv_config.ai_filter_mode,
                display_tag_str=f"随机排行榜:{mode}",
                return_count=self.pixiv_config.return_count,
                logger=logger,
                show_filter_result=self.pixiv_config.show_filter_result,
                excluded_tags=[],
            )

            filtered_illusts, _ = filter_illusts_with_reason(initial_illusts, config)

            if not filtered_illusts:
                logger.info(f"排行榜 {mode} 的随机搜索过滤后无符合条件的作品。")
                return

            illusts_to_send = sample_illusts(
                filtered_illusts, self.pixiv_config.return_count, shuffle=True
            )

            for illust in illusts_to_send:
                await self._send_illust(session_id, illust, chat_id)
                await asyncio.sleep(2)

        except Exception as e:
            logger.error(f"为群组 {chat_id} 执行随机排行榜搜索时出错: {e}")

    def _fix_session_id(self, session_id: str) -> str:
        """
        修复 session_id，将旧格式的平台类型转换为配置的平台实例名称
        例如：aiocqhttp:GroupMessage:123 -> 喵喵ll:GroupMessage:123
        """
        # 如果没有配置平台实例名称，直接返回原始值
        platform_name = self.pixiv_config.platform_instance_name
        if not platform_name:
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
                logger.debug(f"转换 session_id: {session_id} -> {new_session_id}")
                return new_session_id
            
            return session_id
            
        except Exception as e:
            logger.error(f"修复 session_id 时出错: {e}")
            return session_id

    async def _send_illust(self, session_id: str, illust, chat_id: str):
        """发送单个作品到指定会话，支持图片大小限制和自动降级"""
        tmp_path = None
        try:
            # 修复：转换 session_id
            session_id = self._fix_session_id(session_id)
            
            detail_message = build_detail_message(illust, is_novel=False)

            if illust.page_count > 1 and illust.meta_pages:
                url_obj = illust.meta_pages[0].image_urls
            else:
                url_obj = illust.image_urls

            quality_preference = ["original", "large", "medium"]
            start_index = (
                quality_preference.index(self.pixiv_config.image_quality)
                if self.pixiv_config.image_quality in quality_preference
                else 0
            )

            img_data = None
            used_quality = None
            size_limit_mb = self.pixiv_config.image_size_limit_mb
            size_limit_enabled = self.pixiv_config.image_size_limit_enabled

            logger.debug(f"[RandomSearch] PID {illust.id}: size_limit_enabled={size_limit_enabled}, size_limit_mb={size_limit_mb}")

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
                        logger.debug(f"[RandomSearch] PID {illust.id}: 质量 {quality} 无URL，跳过")
                        continue

                    logger.debug(f"[RandomSearch] PID {illust.id}: 尝试下载质量 {quality}")
                    downloaded_data = await download_image(session, image_url)

                    if downloaded_data:
                        size_mb = len(downloaded_data) / (1024 * 1024)
                        logger.debug(f"[RandomSearch] PID {illust.id}: 质量 {quality} 下载成功，大小 {size_mb:.2f}MB")

                        if size_limit_enabled and size_mb > size_limit_mb and quality != "medium":
                            logger.warning(
                                f"图片 PID {illust.id} 质量 {quality} 大小 {size_mb:.2f}MB 超过限制 {size_limit_mb}MB，尝试降级"
                            )
                            continue

                        img_data = downloaded_data
                        used_quality = quality
                        logger.info(f"图片 PID {illust.id} 最终使用质量 {quality}，大小 {size_mb:.2f}MB")
                        break
                    else:
                        logger.debug(f"[RandomSearch] PID {illust.id}: 质量 {quality} 下载失败")

            # 发送消息
            if img_data:
                # 方案1：尝试使用 Image.fromBytes 直接发送
                try:
                    from astrbot.api.message_components import Image, Plain
                    from astrbot.core.message.message_event_result import MessageEventResult
                    
                    logger.debug(f"[RandomSearch] PID {illust.id}: 尝试方案1 - Image.fromBytes")
                    
                    chain_components = [Image.fromBytes(img_data)]
                    if self.pixiv_config.show_details:
                        chain_components.append(Plain(detail_message))
                    
                    result = MessageEventResult()
                    result.chain = chain_components
                    result.use_t2i = False
                    
                    send_result = await self.context.send_message(session_id, result)
                    
                    if send_result:
                        logger.info(f"随机搜索：已发送作品 PID {illust.id} 到 {session_id}")
                        add_sent_illust(illust.id, chat_id)
                        return
                    else:
                        logger.warning(f"[RandomSearch] PID {illust.id}: 方案1 返回 False")
                    
                except Exception as e1:
                    logger.warning(f"[RandomSearch] PID {illust.id}: 方案1失败 - {e1}")
                
                # 方案2：保存到临时文件后使用 file_image
                try:
                    logger.debug(f"[RandomSearch] PID {illust.id}: 尝试方案2 - file_image")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        tmp_file.write(img_data)
                        tmp_path = tmp_file.name
                    
                    logger.debug(f"[RandomSearch] PID {illust.id}: 图片已保存到 {tmp_path}，文件大小 {os.path.getsize(tmp_path)} 字节")
                    
                    if self.pixiv_config.show_details:
                        message_chain = MessageChain().file_image(tmp_path).message(detail_message)
                    else:
                        message_chain = MessageChain().file_image(tmp_path)
                    
                    send_result = await self.context.send_message(session_id, message_chain)
                    
                    if send_result:
                        logger.info(f"随机搜索：已发送作品 PID {illust.id} 到 {session_id}")
                        add_sent_illust(illust.id, chat_id)
                        return
                    else:
                        logger.warning(f"[RandomSearch] PID {illust.id}: 方案2 返回 False")
                    
                except Exception as e2:
                    logger.warning(f"[RandomSearch] PID {illust.id}: 方案2失败 - {e2}")
                
                # 方案3：使用 base64 URI
                try:
                    import base64
                    logger.debug(f"[RandomSearch] PID {illust.id}: 尝试方案3 - base64")
                    
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    base64_uri = f"base64://{img_base64}"
                    
                    message_chain = MessageChain().image(base64_uri)
                    if self.pixiv_config.show_details:
                        message_chain = message_chain.message(detail_message)
                    
                    send_result = await self.context.send_message(session_id, message_chain)
                    
                    if send_result:
                        logger.info(f"随机搜索：已发送作品 PID {illust.id} 到 {session_id}")
                        add_sent_illust(illust.id, chat_id)
                        return
                    else:
                        logger.warning(f"[RandomSearch] PID {illust.id}: 方案3 返回 False")
                    
                except Exception as e3:
                    logger.error(f"[RandomSearch] PID {illust.id}: 方案3失败 - {e3}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                # 所有方案都失败，仅发送文本
                logger.error(f"[RandomSearch] PID {illust.id}: 所有发送方案都失败，仅发送文本")
                message_chain = MessageChain().message(f"[图片发送失败]\n{detail_message}")
                await self.context.send_message(session_id, message_chain)
                add_sent_illust(illust.id, chat_id)
                
            else:
                logger.warning(f"[RandomSearch] PID {illust.id}: 所有质量下载失败，仅发送文本")
                message_chain = MessageChain().message(f"[图片下载失败或过大]\n{detail_message}")
                await self.context.send_message(session_id, message_chain)

        except Exception as e:
            logger.error(f"发送作品 PID {illust.id} 时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # 清理临时文件
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                    logger.debug(f"[RandomSearch] PID {illust.id}: 临时文件已清理")
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {e}")

    def suspend_group_search(self, chat_id: str):
        """暂停指定群组的随机搜索"""
        try:
            remove_schedule_time(chat_id)
            logger.info(f"已移除群组 {chat_id} 的调度时间")
        except Exception as e:
            logger.error(f"移除群组 {chat_id} 调度时间失败: {e}")

    def resume_group_search(self, chat_id: str):
        """恢复指定群组的随机搜索"""
        try:
            now = datetime.now()
            min_interval = self.pixiv_config.random_search_min_interval
            max_interval = self.pixiv_config.random_search_max_interval
            if max_interval < min_interval:
                max_interval = min_interval

            delay_minutes = random.randint(min_interval, max_interval)
            next_time = now + timedelta(minutes=delay_minutes)
            set_schedule_time(chat_id, next_time)
            logger.info(f"群组 {chat_id} 随机搜索已恢复，将在 {delay_minutes} 分钟后执行")
        except Exception as e:
            logger.error(f"恢复群组 {chat_id} 调度时间失败: {e}")

    def get_queue_status(self) -> dict:
        """获取队列状态信息，用于调试和监控"""
        return {
            "queue_size": self.task_queue.qsize(),
            "is_queue_processor_running": self.is_queue_processor_running,
            "execution_locks": dict(self.execution_locks),
            "active_groups": [
                chat_id for chat_id, locked in self.execution_locks.items() if locked
            ],
        }

    async def force_execute_group(self, chat_id: str) -> bool:
        """强制执行指定群组的随机搜索（用于调试）"""
        if chat_id not in self.execution_locks:
            self.execution_locks[chat_id] = False

        if self.execution_locks[chat_id]:
            logger.warning(f"群组 {chat_id} 已在执行状态，无法强制执行")
            return False

        try:
            await self.task_queue.put(chat_id)
            logger.info(f"群组 {chat_id} 已强制加入执行队列")
            return True
        except Exception as e:
            logger.error(f"强制执行群组 {chat_id} 失败: {e}")
            return False

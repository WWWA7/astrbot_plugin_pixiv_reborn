import asyncio
import random
import os
import tempfile
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from astrbot.api import logger
from astrbot.core.message.message_event_result import MessageChain, MessageEventResult
from astrbot.api.message_components import Image, Plain

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
from .pixiv_utils import download_illust_all_pages, build_page_hint
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
                       f"max_pages_per_illust={self.pixiv_config.max_pages_per_illust}")
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
        """检查是否有群组需要执行搜索"""
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
        """任务队列处理器"""
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
        """定期清理过期记录"""
        try:
            logger.info("开始清理过期的已发送作品记录...")
            days = self.pixiv_config.random_sent_illust_retention_days
            await asyncio.to_thread(cleanup_old_sent_illusts, days=days)
            logger.info("清理过期记录任务完成。")
        except Exception as e:
            logger.error(f"清理过期记录任务出错: {e}")

    async def execute_search_for_group(self, chat_id: str):
        """为特定群组执行随机搜索"""
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
                else:
                    break

                next_url = json_result.next_url
                next_params = self.client.parse_qs(next_url) if next_url else None

                if next_params:
                    await asyncio.sleep(0.5)

            if not all_illusts:
                logger.info(f"标签 {raw_tag} 的随机搜索未返回结果。")
                return

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

        logger.info(f"正在为群组 {chat_id} 执行随机排行榜搜索，模式: {mode}")

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
        """修复 session_id"""
        platform_name = self.pixiv_config.platform_instance_name
        if not platform_name:
            return session_id
        
        try:
            parts = session_id.split(":")
            if len(parts) < 3:
                return session_id
            
            platform_identifier = parts[0]
            old_platform_types = [
                "aiocqhttp", "onebot", "cqhttp", "gocqhttp", 
                "napcat", "llonebot", "shamrock"
            ]
            
            if platform_identifier.lower() in old_platform_types:
                new_session_id = f"{platform_name}:{':'.join(parts[1:])}"
                logger.debug(f"转换 session_id: {session_id} -> {new_session_id}")
                return new_session_id
            
            return session_id
        except Exception as e:
            logger.error(f"修复 session_id 时出错: {e}")
            return session_id

    async def _send_illust(self, session_id: str, illust, chat_id: str):
        """发送单个作品，支持多页"""
        try:
            session_id = self._fix_session_id(session_id)
            
            detail_message = build_detail_message(illust, is_novel=False)
            max_pages = self.pixiv_config.max_pages_per_illust if self.pixiv_config.max_pages_per_illust > 0 else 0

            async with aiohttp.ClientSession() as session:
                images_data, sent_pages, total_pages = await download_illust_all_pages(session, illust, max_pages)

            if not images_data:
                logger.warning(f"[RandomSearch] PID {illust.id}: 所有质量下载失败，仅发送文本")
                message_chain = MessageChain().message(f"[图片下载失败]\n{detail_message}")
                await self.context.send_message(session_id, message_chain)
                return

            # 添加页数提示
            page_hint = build_page_hint(sent_pages, total_pages)
            final_message = detail_message + page_hint

            # 构建消息
            result_obj = MessageEventResult()
            result_obj.chain = [Image.fromBytes(img_data) for img_data in images_data]
            if self.pixiv_config.show_details:
                result_obj.chain.append(Plain(final_message))
            result_obj.use_t2i = False

            send_result = await self.context.send_message(session_id, result_obj)

            if send_result:
                logger.info(f"随机搜索：已发送作品 PID {illust.id} ({sent_pages}/{total_pages}页) 到 {session_id}")
                add_sent_illust(illust.id, chat_id)
            else:
                logger.warning(f"[RandomSearch] PID {illust.id}: send_message 返回 False")

        except Exception as e:
            logger.error(f"发送作品 PID {illust.id} 时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())

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
        """获取队列状态信息"""
        return {
            "queue_size": self.task_queue.qsize(),
            "is_queue_processor_running": self.is_queue_processor_running,
            "execution_locks": dict(self.execution_locks),
            "active_groups": [
                chat_id for chat_id, locked in self.execution_locks.items() if locked
            ],
        }

    async def force_execute_group(self, chat_id: str) -> bool:
        """强制执行指定群组的随机搜索"""
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

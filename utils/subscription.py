import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from astrbot.api import logger
from astrbot.core.message.message_event_result import MessageChain, MessageEventResult
from astrbot.api.message_components import Image, Plain, Node, Nodes
from pixivpy3 import AppPixivAPI
from ..utils.pixiv_utils import (
    filter_items,
    download_illust_all_pages,
    build_page_hint,
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
        
        log_dir = r"C:\Users\Administrator\Desktop\AstrBot\pixiv"
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self.log_file = os.path.join(log_dir, "pixiv_subscription.log")
        
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
                          f"max_pages_per_illust={self.pixiv_config.max_pages_per_illust}, "
                          f"forward_threshold={self.pixiv_config.forward_threshold}, "
                          f"platform_instance_name={self.pixiv_config.platform_instance_name}")
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
                return new_session_id
            
            return session_id
        except Exception as e:
            self._save_log(f"[ERROR] 修复 session_id 时出错: {e}")
            return session_id

    async def check_subscriptions(self):
        """检查所有订阅并推送更新"""
        self._save_log("开始新一轮订阅检查...")
        
        if not await self.client_wrapper.authenticate():
            self._save_log("订阅检查失败：Pixiv API 认证失败。")
            return

        subscriptions = get_all_subscriptions()
        if not subscriptions:
            self._save_log("当前无任何订阅记录。")
            return

        self._save_log(f"获取到 {len(subscriptions)} 条订阅记录，开始逐个检查。")

        # 收集所有新作品，按 session_id 分组
        updates_by_session = defaultdict(list)
        artist_updates_by_session = defaultdict(lambda: defaultdict(int))

        for sub in subscriptions:
            try:
                if sub.sub_type == "artist":
                    new_illusts = await self._check_artist_updates(sub)
                    if new_illusts:
                        session_id = self._fix_session_id(sub.session_id)
                        for illust in new_illusts:
                            updates_by_session[session_id].append((sub, illust))
                            artist_updates_by_session[session_id][sub.target_name] += 1
            except Exception as e:
                self._save_log(f"检查订阅 {sub.sub_type}: {sub.target_id} 时发生错误: {e}")
                import traceback
                self._save_log(traceback.format_exc())
            await asyncio.sleep(2)

        # 按 session_id 发送更新
        for session_id, updates in updates_by_session.items():
            if not updates:
                continue
            
            artist_counts = artist_updates_by_session[session_id]
            total_count = len(updates)
            
            self._save_log(f"准备向 {session_id} 发送 {total_count} 张新作品")
            
            if self.pixiv_config.forward_threshold:
                await self._send_updates_forward(session_id, updates, artist_counts)
            else:
                await self._send_updates_normal(session_id, updates)

        self._save_log("本轮订阅检查完成。")

    async def _check_artist_updates(self, sub) -> list:
        """检查画师更新，返回新作品列表"""
        api: AppPixivAPI = self.client
        json_result = await asyncio.to_thread(api.user_illusts, sub.target_id)

        if not json_result or not json_result.illusts:
            return []

        new_illusts = []
        for illust in json_result.illusts:
            if illust.id > sub.last_notified_illust_id:
                new_illusts.append(illust)
            else:
                break

        if not new_illusts:
            return []

        new_ids = [str(i.id) for i in new_illusts]
        self._save_log(f"画师 [{sub.target_name}] 发现 {len(new_illusts)} 张新作品! IDs: {', '.join(new_ids)}")
        
        new_illusts.reverse()
        latest_id = new_illusts[-1].id
        update_last_notified_id(sub.chat_id, sub.sub_type, sub.target_id, latest_id)
        self._save_log(f"更新数据库 last_id -> {latest_id}")

        # 过滤作品
        filtered_illusts = []
        for illust in new_illusts:
            filtered, reason = filter_items([illust], f"画师订阅: {sub.target_name}")
            if filtered:
                filtered_illusts.append(filtered[0])
            else:
                self._save_log(f"作品 PID: {illust.id} 被过滤，原因: {reason}")

        return filtered_illusts

    async def _send_updates_forward(self, session_id: str, updates: list, artist_counts: dict):
        """转发模式：先发送提示消息，再发送合并的转发消息"""
        import aiohttp
        
        try:
            # 1. 构建并发送提示消息
            total_count = len(updates)
            artist_info_list = [f"{name} ({count}张)" for name, count in artist_counts.items()]
            artist_info_str = "、".join(artist_info_list)
            
            notify_message = f"您订阅的画师有新作品更新啦！\n画师: {artist_info_str}\n共 {total_count} 张新作品"
            
            notify_chain = MessageChain().message(notify_message)
            result = await self.context.send_message(session_id, notify_chain)
            
            if result:
                self._save_log(f"提示消息发送成功")
            else:
                self._save_log(f"[WARN] 提示消息发送失败")
            
            await asyncio.sleep(1)
            
            # 2. 构建转发消息
            nodes_list = []
            nickname = "Pixiv订阅"
            max_pages = self.pixiv_config.max_pages_per_illust if self.pixiv_config.max_pages_per_illust > 0 else 0
            
            async with aiohttp.ClientSession() as session:
                for sub, illust in updates:
                    # 构建详情消息
                    detail_message = f"画师: {sub.target_name}\n"
                    detail_message += build_detail_message(illust, is_novel=False)
                    
                    # 下载多页图片
                    images_data, sent_pages, total_pages = await download_illust_all_pages(session, illust, max_pages)
                    
                    node_content = []
                    
                    if images_data:
                        for img_data in images_data:
                            node_content.append(Image.fromBytes(img_data))
                        
                        # 添加页数提示
                        page_hint = build_page_hint(sent_pages, total_pages)
                        final_message = detail_message + page_hint
                        
                        if self.pixiv_config.show_details:
                            node_content.append(Plain(final_message))
                    else:
                        node_content = [Plain(f"[图片下载失败]\n{detail_message}")]
                    
                    nodes_list.append(Node(name=nickname, content=node_content))
                    self._save_log(f"[DEBUG] 已添加作品 PID: {illust.id} ({sent_pages}/{total_pages}页) 到转发列表")
            
            if nodes_list:
                forward_result = MessageEventResult()
                forward_result.chain = [Nodes(nodes=nodes_list)]
                forward_result.use_t2i = False
                
                send_result = await self.context.send_message(session_id, forward_result)
                
                if send_result:
                    self._save_log(f"转发消息发送成功: 共 {len(nodes_list)} 张作品")
                else:
                    self._save_log(f"[ERROR] 转发消息发送失败: send_message 返回 False")
            else:
                self._save_log(f"[WARN] 没有可发送的作品")
                
        except Exception as e:
            self._save_log(f"转发模式发送失败: {e}")
            import traceback
            self._save_log(traceback.format_exc())

    async def _send_updates_normal(self, session_id: str, updates: list):
        """普通模式：逐张发送"""
        for sub, illust in updates:
            await self._send_single_update(session_id, sub, illust)
            await asyncio.sleep(2)

    async def _send_single_update(self, session_id: str, sub, illust):
        """发送单张作品（普通模式），支持多页"""
        import aiohttp
        
        try:
            detail_message = f"您订阅的画师 [{sub.target_name}] 有新作品啦！\n"
            detail_message += build_detail_message(illust, is_novel=False)

            max_pages = self.pixiv_config.max_pages_per_illust if self.pixiv_config.max_pages_per_illust > 0 else 0

            async with aiohttp.ClientSession() as session:
                images_data, sent_pages, total_pages = await download_illust_all_pages(session, illust, max_pages)

            if not images_data:
                self._save_log(f"[ERROR] PID {illust.id}: 所有质量下载失败")
                return

            # 添加页数提示
            page_hint = build_page_hint(sent_pages, total_pages)
            final_message = detail_message + page_hint

            # 构建消息：多张图片 + 详情
            result_obj = MessageEventResult()
            result_obj.chain = [Image.fromBytes(img_data) for img_data in images_data]
            if self.pixiv_config.show_details:
                result_obj.chain.append(Plain(final_message))
            result_obj.use_t2i = False
            
            result = await self.context.send_message(session_id, result_obj)
            
            if result:
                self._save_log(f"推送成功: PID {illust.id} ({sent_pages}/{total_pages}页)")
            else:
                self._save_log(f"[ERROR] PID {illust.id}: send_message 返回 False")

        except Exception as e:
            self._save_log(f"发送订阅更新时出错 (PID {illust.id}): {e}")
            import traceback
            self._save_log(traceback.format_exc())

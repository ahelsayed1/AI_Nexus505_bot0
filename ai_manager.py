# ai_manager.py - النسخة المعدلة (OpenAI أولويات جديدة + Veo 3.1)
# الإصدار: 5.4 (Smart Multi-Source AI - OpenAI Priority Update + Veo 3.1)
import os
import logging
import asyncio
import google.generativeai as genai
from openai import OpenAI as OpenAIClient
import aiohttp
import re
import json
import base64
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from urllib.parse import urlparse
import hashlib
from collections import OrderedDict
import time
import uuid

logger = logging.getLogger(__name__)

class ServiceType(Enum):
    """أنواع الخدمات المتاحة"""
    CHAT = "chat"
    IMAGE = "image"
    VIDEO = "video"

class Provider(Enum):
    """مزودي الخدمات"""
    GOOGLE = "google"
    OPENAI = "openai"
    STABILITY = "stability"
    LUMA = "luma"
    KLING = "kling"

@dataclass
class ModelInfo:
    """معلومات الموديل"""
    name: str
    provider: Provider
    service_type: ServiceType
    version: str = "1.0"
    release_date: Optional[str] = None
    max_tokens: int = 2048
    is_latest: bool = False
    is_deprecated: bool = False
    priority: int = 100
    supports_enhancement: bool = True

@dataclass
class ProviderConfig:
    """إعدادات مزود الخدمة"""
    name: Provider
    api_key: Optional[str] = None
    enabled: bool = False
    daily_limit: int = 100
    usage_today: int = 0
    errors_today: int = 0
    avg_response_time: float = 0.0
    last_error: Optional[str] = None
    
    discovered_models: Dict[ServiceType, List[ModelInfo]] = field(default_factory=dict)
    active_models: Dict[ServiceType, str] = field(default_factory=dict)


# ==================== نظام Rate Limiting ====================

class RateLimiter:
    """نظام تحديد معدل الطلبات (Rate Limiting)"""
    
    def __init__(self):
        self.user_requests = {}
        self.default_limits = {
            "ai_chat": {"per_second": 1, "per_minute": 20},
            "image_gen": {"per_second": 0.5, "per_minute": 5},
            "video_gen": {"per_second": 0.1, "per_minute": 2}
        }
        self.window_size = 60
        self.lock = asyncio.Lock()
        self.cleanup_task = None
        logger.info("🚦 تم تهيئة نظام Rate Limiting")
    
    async def start_cleanup_task(self):
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        while True:
            await asyncio.sleep(60)
            await self._cleanup_old_records()
    
    async def _cleanup_old_records(self):
        async with self.lock:
            now = time.time()
            cutoff = now - self.window_size
            users_to_delete = []
            for user_id, data in self.user_requests.items():
                data["timestamps"] = [ts for ts in data["timestamps"] if ts > cutoff]
                if not data["timestamps"]:
                    users_to_delete.append(user_id)
            for user_id in users_to_delete:
                del self.user_requests[user_id]
    
    async def check_rate_limit(self, user_id: int, service_type: str) -> Tuple[bool, str]:
        try:
            if self.cleanup_task is None:
                await self.start_cleanup_task()
            
            limits = self.default_limits.get(service_type)
            if not limits:
                return True, ""
            
            async with self.lock:
                now = time.time()
                if user_id not in self.user_requests:
                    self.user_requests[user_id] = {
                        "timestamps": [],
                        "count_per_minute": 0,
                        "last_reset": now
                    }
                
                user_data = self.user_requests[user_id]
                if now - user_data["last_reset"] >= 60:
                    user_data["count_per_minute"] = 0
                    user_data["last_reset"] = now
                
                cutoff = now - self.window_size
                user_data["timestamps"] = [ts for ts in user_data["timestamps"] if ts > cutoff]
                
                if limits.get("per_second", 0) > 0:
                    second_cutoff = now - 1
                    requests_last_second = sum(1 for ts in user_data["timestamps"] if ts > second_cutoff)
                    if requests_last_second >= limits["per_second"]:
                        wait_time = 1.0 - (now - max(user_data["timestamps"][-1] if user_data["timestamps"] else now, now - 1))
                        return False, f"⏳ أنت ترسل بسرعة كبيرة. انتظر {wait_time:.1f} ثانية."
                
                if limits.get("per_minute", 0) > 0:
                    if user_data["count_per_minute"] >= limits["per_minute"]:
                        reset_time = 60 - (now - user_data["last_reset"])
                        return False, f"⏳ لقد استهلكت رصيد الدقيقة. انتظر {reset_time:.0f} ثانية."
                
                user_data["timestamps"].append(now)
                user_data["count_per_minute"] += 1
                if len(user_data["timestamps"]) > 100:
                    user_data["timestamps"] = user_data["timestamps"][-50:]
                return True, ""
        except Exception as e:
            logger.error(f"❌ خطأ في Rate Limiter: {e}")
            return True, ""
    
    async def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        async with self.lock:
            if user_id not in self.user_requests:
                return {"requests_last_minute": 0, "reset_in": 0}
            user_data = self.user_requests[user_id]
            now = time.time()
            return {
                "requests_last_minute": len([ts for ts in user_data["timestamps"] if ts > now - 60]),
                "reset_in": max(0, 60 - (now - user_data["last_reset"])),
                "total_tracked": len(user_data["timestamps"])
            }


class SmartAIManager:
    """
    مدير ذكاء اصطناعي ذكي يكتشف الموديلات تلقائياً
    ويرتبها من الأحدث إلى الأقدم مع دعم كامل للخدمات
    """
    
    def __init__(self, db):
        """تهيئة المدير الذكي"""
        self.db = db
        self.user_limits_cache = OrderedDict()
        self.max_cache_size = 1000
        self.rate_limiter = RateLimiter()
        
        self.chat_sessions: Dict[int, Dict[str, Any]] = {}
        self.session_timeout = timedelta(hours=1)
        self.generated_files_cache: Dict[str, Dict] = {}
        
        # ========== (تعديل هام) نقلنا إعدادات جوجل للأول ==========
        self.google_project_id = os.getenv("GOOGLE_PROJECT_ID")
        self.google_location = os.getenv("GOOGLE_LOCATION", "us-central1")
        self.google_imagen_model = "imagen-3.0-generate-001"
        
        # ========== إعدادات Google Veo (مفعلة بالكامل) ==========
        self.veo_api_key = os.getenv("GOOGLE_AI_API_KEY")
        self.veo_project_id = self.google_project_id
        self.veo_location = self.google_location
        
        self.discovery_completed = False
        self.discovery_lock = asyncio.Lock()
        self.default_timeout = aiohttp.ClientTimeout(total=30)
        
        # ========== دلوقتي نستدعي المزودين بأمان ==========
        self.providers: Dict[Provider, ProviderConfig] = self._init_providers()
        
        logger.info("🚀 تم تهيئة النظام الذكي (نسخة OpenAI محدثة + Veo 3.1)")
    
    def _init_providers(self) -> Dict[Provider, ProviderConfig]:
        """تهيئة إعدادات جميع المزودين"""
        providers = {
            Provider.GOOGLE: ProviderConfig(
                name=Provider.GOOGLE,
                api_key=os.getenv("GOOGLE_AI_API_KEY"),
                daily_limit=int(os.getenv("GOOGLE_DAILY_LIMIT", "50"))
            ),
            Provider.OPENAI: ProviderConfig(
                name=Provider.OPENAI,
                api_key=os.getenv("OPENAI_API_KEY"),
                daily_limit=int(os.getenv("OPENAI_DAILY_LIMIT", "30"))
            ),
            Provider.STABILITY: ProviderConfig(
                name=Provider.STABILITY,
                api_key=os.getenv("STABILITY_API_KEY"),
                daily_limit=int(os.getenv("STABILITY_DAILY_LIMIT", "20"))
            ),
            Provider.LUMA: ProviderConfig(
                name=Provider.LUMA,
                api_key=os.getenv("LUMAAI_API_KEY"),
                daily_limit=int(os.getenv("LUMA_DAILY_LIMIT", "10"))
            ),
            Provider.KLING: ProviderConfig(
                name=Provider.KLING,
                api_key=os.getenv("KLING_API_KEY"),
                daily_limit=int(os.getenv("KLING_DAILY_LIMIT", "5"))
            )
        }
        
        for provider_name, config in providers.items():
            if not config.api_key or config.api_key.strip() == "":
                config.enabled = False
                logger.warning(f"⚠️ {provider_name.value}: مفتاح API غير موجود")
            elif len(config.api_key) < 10:
                config.enabled = False
                logger.warning(f"⚠️ {provider_name.value}: مفتاح API قصير جداً")
            else:
                if provider_name == Provider.GOOGLE and not self.google_project_id:
                    logger.warning(f"⚠️ Google: GOOGLE_PROJECT_ID غير موجود - Imagen/Veo سيكون غير متاح")
        
        return providers
    
    async def ensure_discovery(self):
        if not self.discovery_completed:
            async with self.discovery_lock:
                if not self.discovery_completed:
                    await self._setup_and_discover_async()
                    self.discovery_completed = True
    
    async def _setup_and_discover_async(self):
        try:
            logger.info("🔍 بدء اكتشاف الموديلات...")
            await self._setup_and_discover_google()
            await self._setup_and_discover_openai()  # OpenAI بالأولويات الجديدة
            await self._setup_other_apis()
            self._log_discovery_results()
            self._select_best_models()
            await self.rate_limiter.start_cleanup_task()
            logger.info("✅ اكتمل اكتشاف الموديلات")
        except Exception as e:
            logger.error(f"❌ فشل اكتشاف الموديلات: {e}", exc_info=True)
    
    # ==================== GOOGLE ====================
    
    async def _setup_and_discover_google(self):
        google_config = self.providers[Provider.GOOGLE]
        if not google_config.api_key:
            logger.warning("⚠️ Google API Key غير موجود")
            return
        
        try:
            genai.configure(api_key=google_config.api_key)
            google_config.enabled = True
            logger.info("🔍 جاري اكتشاف موديلات Google...")
            
            prioritized_models = [
                # محادثة - Gemini
                'gemini-2.5-flash', 'gemini-2.5-pro',
                'gemini-3-flash-preview', 'gemini-3-pro-preview',
                'gemini-2.0-flash', 'gemini-1.5-pro',
                'gemini-1.5-flash', 'gemini-1.0-pro',
                
                # صور - Imagen
                'nano-banana-pro-preview',
                'imagen-4.0-generate-preview-06-06',
                'imagen-3.0-generate-001',
                
                # ========== فيديو - Veo (مفعل بالكامل) ==========
                'veo-3.1-generate-001',      # Veo 3.1 (أحدث إصدار)
                'veo-3.1-fast-generate-001',  # Veo 3.1 Fast (أسرع)
                'veo-3.0-generate-001'        # Veo 3.0 (للتوافق)
            ]
            
            google_config.discovered_models = {
                ServiceType.CHAT: [],
                ServiceType.IMAGE: [],
                ServiceType.VIDEO: []  # هنضيف Veo هنا
            }
            
            for model_name in prioritized_models:
                model_info = self._analyze_google_model(model_name)
                if model_info:
                    google_config.discovered_models[model_info.service_type].append(model_info)
            
            # ترتيب حسب الأولوية
            for service_type in google_config.discovered_models:
                google_config.discovered_models[service_type].sort(key=lambda x: x.priority)
            
            # تفعيل الموديلات
            if google_config.discovered_models[ServiceType.CHAT]:
                top_chat = google_config.discovered_models[ServiceType.CHAT][0]
                google_config.active_models[ServiceType.CHAT] = top_chat.name
                logger.info(f"👑 Gemini: {top_chat.name}")
            
            # تفعيل Veo تلقائياً
            if google_config.discovered_models[ServiceType.VIDEO]:
                top_video = google_config.discovered_models[ServiceType.VIDEO][0]
                google_config.active_models[ServiceType.VIDEO] = top_video.name
                logger.info(f"🎬 Veo نشط: {top_video.name} (Priority: {top_video.priority})")
            
        except Exception as e:
            logger.error(f"❌ فشل إعداد Google: {e}")
            google_config.enabled = False

    def _analyze_google_model(self, model_name: str) -> Optional[ModelInfo]:
        """تحليل موديل Google"""
        try:
            model_lower = model_name.lower()
            
            if 'gemini' in model_lower:
                service_type = ServiceType.CHAT
            elif 'imagen' in model_lower or 'banana' in model_lower:
                service_type = ServiceType.IMAGE
            elif 'veo' in model_lower:
                service_type = ServiceType.VIDEO  # Veo للفيديو
            else:
                return None
            
            version = "1.0"
            priority = 100
            
            if service_type == ServiceType.CHAT:
                if 'gemini-2.5-flash' in model_lower:
                    version = "2.5"; priority = 10
                elif 'gemini-2.5-pro' in model_lower:
                    version = "2.5"; priority = 15
                elif 'gemini-3-flash-preview' in model_lower or 'gemini-3-pro-preview' in model_lower:
                    version = "3.0"; priority = 10
                elif 'gemini-2.0-flash' in model_lower:
                    version = "2.0"; priority = 20
                elif 'gemini-1.5-pro' in model_lower:
                    version = "1.5"; priority = 30
                elif 'gemini-1.5-flash' in model_lower:
                    version = "1.5"; priority = 30
                elif 'gemini-1.0-pro' in model_lower:
                    version = "1.0"; priority = 40
                
            elif service_type == ServiceType.IMAGE:
                if 'banana' in model_lower:
                    priority = 5
                elif 'imagen-4' in model_lower:
                    priority = 10
                elif 'imagen-3' in model_lower:
                    priority = 20
                elif 'imagen-2' in model_lower:
                    priority = 30
                else:
                    priority = 50
                
            elif service_type == ServiceType.VIDEO:
                # ========== Veo 3.1 الأولويات ==========
                if 'veo-3.1-fast' in model_lower:
                    version = "3.1-fast"
                    priority = 5  # الأعلى أولوية (أسرع)
                elif 'veo-3.1' in model_lower:
                    version = "3.1"
                    priority = 10  # الجودة العالية
                elif 'veo-3.0' in model_lower:
                    version = "3.0"
                    priority = 20  # للتوافق
                else:
                    priority = 30
            
            return ModelInfo(
                name=model_name,
                provider=Provider.GOOGLE,
                service_type=service_type,
                version=version,
                is_latest=('preview' in model_lower or 'latest' in model_lower or 'fast' in model_lower),
                priority=priority
            )
            
        except Exception as e:
            logger.debug(f"⚠️ فشل تحليل موديل Google {model_name}: {e}")
            return None
    
    # ==================== OPENAI - أولويات جديدة حسب طلبك ====================
    
    async def _setup_and_discover_openai(self):
        """إعداد OpenAI API - مع أولويات GPT-5.0, GPT-5.2, GPT-4.5, GPT-4.1"""
        openai_config = self.providers[Provider.OPENAI]
        if not openai_config.api_key:
            logger.warning("⚠️ OpenAI API Key غير موجود")
            return
        
        try:
            self.openai_client = OpenAIClient(api_key=openai_config.api_key)
            openai_config.enabled = True
            logger.info("🔍 جاري اكتشاف موديلات OpenAI...")
            
            openai_config.discovered_models = {
                ServiceType.CHAT: [],
                ServiceType.IMAGE: [],
                ServiceType.VIDEO: []
            }
            
            # ========== المحادثة (Chat) - الأولويات الجديدة ==========
            chat_models = [
                # 1. GPT-5.0 (الأحدث - Priority 1)
                'gpt-5.0-turbo', 'gpt-5.0',
                
                # 2. GPT-5.2 (Priority 2)
                'gpt-5.2-turbo', 'gpt-5.2',
                
                # 3. GPT-4.5 (Priority 3)
                'gpt-4.5-turbo', 'gpt-4.5',
                
                # 4. GPT-4.1 (Priority 4)
                'gpt-4.1-turbo', 'gpt-4.1',
                
                # 5. الباقي (أولوية أقل)
                'gpt-4o', 'gpt-4o-mini',
                'gpt-4-turbo', 'gpt-4',
                'gpt-3.5-turbo', 'gpt-3.5-turbo-instruct'
            ]
            
            for model_name in chat_models:
                model_info = self._analyze_openai_chat_model(model_name)
                if model_info:
                    openai_config.discovered_models[ServiceType.CHAT].append(model_info)
            
            # ========== الصور (Images) - مع GPT-Image-1.5 ==========
            image_models = [
                # 1. GPT-Image-1.5 (Priority 1)
                'gpt-image-1.5',
                
                # 2. DALL-E 3 (Priority 5)
                'dall-e-3',
                
                # 3. DALL-E 2 (Priority 10)
                'dall-e-2'
            ]
            
            for model_name in image_models:
                model_info = self._analyze_openai_image_model(model_name)
                if model_info:
                    openai_config.discovered_models[ServiceType.IMAGE].append(model_info)
            
            # ترتيب الموديلات حسب الأولوية
            openai_config.discovered_models[ServiceType.CHAT].sort(key=lambda x: x.priority)
            openai_config.discovered_models[ServiceType.IMAGE].sort(key=lambda x: x.priority)
            
            # تفعيل أفضل الموديلات
            if openai_config.discovered_models[ServiceType.CHAT]:
                best_chat = openai_config.discovered_models[ServiceType.CHAT][0]
                openai_config.active_models[ServiceType.CHAT] = best_chat.name
                logger.info(f"🤖 OpenAI Chat: {best_chat.name} (Priority: {best_chat.priority})")
            
            if openai_config.discovered_models[ServiceType.IMAGE]:
                best_image = openai_config.discovered_models[ServiceType.IMAGE][0]
                openai_config.active_models[ServiceType.IMAGE] = best_image.name
                logger.info(f"🎨 OpenAI Image: {best_image.name} (Priority: {best_image.priority})")
            
        except Exception as e:
            logger.error(f"❌ فشل إعداد OpenAI: {e}")
            openai_config.enabled = False
    
    def _analyze_openai_chat_model(self, model_name: str) -> Optional[ModelInfo]:
        """تحليل موديل محادثة OpenAI - بالأولويات الجديدة"""
        try:
            model_lower = model_name.lower()
            
            # ========== نظام الأولويات الجديد ==========
            if 'gpt-5.0' in model_lower:
                return ModelInfo(
                    name=model_name,
                    provider=Provider.OPENAI,
                    service_type=ServiceType.CHAT,
                    version="5.0",
                    priority=1,  # الأعلى أولوية
                    is_latest=True
                )
            elif 'gpt-5.2' in model_lower:
                return ModelInfo(
                    name=model_name,
                    provider=Provider.OPENAI,
                    service_type=ServiceType.CHAT,
                    version="5.2",
                    priority=2,
                    is_latest=True
                )
            elif 'gpt-4.5' in model_lower:
                return ModelInfo(
                    name=model_name,
                    provider=Provider.OPENAI,
                    service_type=ServiceType.CHAT,
                    version="4.5",
                    priority=3
                )
            elif 'gpt-4.1' in model_lower:
                return ModelInfo(
                    name=model_name,
                    provider=Provider.OPENAI,
                    service_type=ServiceType.CHAT,
                    version="4.1",
                    priority=4
                )
            elif 'gpt-4o' in model_lower:
                priority = 5 if 'mini' in model_lower else 10
                return ModelInfo(
                    name=model_name,
                    provider=Provider.OPENAI,
                    service_type=ServiceType.CHAT,
                    version="4.0",
                    priority=priority
                )
            elif 'gpt-4-turbo' in model_lower:
                return ModelInfo(
                    name=model_name,
                    provider=Provider.OPENAI,
                    service_type=ServiceType.CHAT,
                    version="4.0",
                    priority=15
                )
            elif 'gpt-4' in model_lower:
                return ModelInfo(
                    name=model_name,
                    provider=Provider.OPENAI,
                    service_type=ServiceType.CHAT,
                    version="4.0",
                    priority=20
                )
            elif 'gpt-3.5-turbo' in model_lower:
                priority = 30 if 'instruct' in model_lower else 25
                return ModelInfo(
                    name=model_name,
                    provider=Provider.OPENAI,
                    service_type=ServiceType.CHAT,
                    version="3.5",
                    priority=priority
                )
            
            return None
            
        except Exception:
            return None
    
    def _analyze_openai_image_model(self, model_name: str) -> Optional[ModelInfo]:
        """تحليل موديل صور OpenAI - مع GPT-Image-1.5"""
        try:
            model_lower = model_name.lower()
            
            if 'gpt-image-1.5' in model_lower:
                return ModelInfo(
                    name=model_name,
                    provider=Provider.OPENAI,
                    service_type=ServiceType.IMAGE,
                    version="1.5",
                    priority=1,  # الأعلى أولوية
                    is_latest=True
                )
            elif 'dall-e-3' in model_lower:
                return ModelInfo(
                    name=model_name,
                    provider=Provider.OPENAI,
                    service_type=ServiceType.IMAGE,
                    version="3.0",
                    priority=5
                )
            elif 'dall-e-2' in model_lower:
                return ModelInfo(
                    name=model_name,
                    provider=Provider.OPENAI,
                    service_type=ServiceType.IMAGE,
                    version="2.0",
                    priority=10
                )
            
            return None
            
        except Exception:
            return None
    
    # ==================== STABILITY & OTHER APIS ====================
    
    async def _setup_other_apis(self):
        """إعداد باقي APIs"""
        
        # ========== STABILITY AI ==========
        stability_config = self.providers[Provider.STABILITY]
        if stability_config.api_key and len(stability_config.api_key) > 20:
            stability_config.enabled = True
            self.stability_headers = {
                "Authorization": f"Bearer {stability_config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            stability_config.discovered_models[ServiceType.IMAGE] = [
                ModelInfo(name="stable-diffusion-3.5-large", provider=Provider.STABILITY,
                         service_type=ServiceType.IMAGE, version="3.5", priority=5, is_latest=True),
                ModelInfo(name="stable-diffusion-xl-1024-v1-0", provider=Provider.STABILITY,
                         service_type=ServiceType.IMAGE, version="XL 1.0", priority=6),
                ModelInfo(name="stable-diffusion-3.5-medium", provider=Provider.STABILITY,
                         service_type=ServiceType.IMAGE, version="3.5", priority=10),
            ]
            
            if stability_config.discovered_models[ServiceType.IMAGE]:
                best_model = min(stability_config.discovered_models[ServiceType.IMAGE], key=lambda x: x.priority)
                stability_config.active_models[ServiceType.IMAGE] = best_model.name
                logger.info(f"✅ Stability AI: {best_model.name}")
        
        # ========== Luma AI ==========
        luma_config = self.providers[Provider.LUMA]
        if luma_config.api_key and len(luma_config.api_key) > 20:
            luma_config.enabled = True
            self.luma_headers = {
                "Authorization": f"Bearer {luma_config.api_key}",
                "Content-Type": "application/json"
            }
            luma_model = ModelInfo(name="dream-machine", provider=Provider.LUMA,
                                  service_type=ServiceType.VIDEO, version="1.0", priority=10)
            luma_config.discovered_models[ServiceType.VIDEO] = [luma_model]
            luma_config.active_models[ServiceType.VIDEO] = "dream-machine"
            logger.info("✅ Luma AI مفعل")
        
        # ========== Kling AI ==========
        kling_config = self.providers[Provider.KLING]
        if kling_config.api_key and len(kling_config.api_key) > 20:
            kling_config.enabled = True
            self.kling_headers = {
                "Authorization": f"Bearer {kling_config.api_key}",
                "Content-Type": "application/json"
            }
            logger.info("✅ Kling AI مفعل")
    
    def _log_discovery_results(self):
        """تسجيل نتائج الاكتشاف"""
        logger.info("=" * 60)
        logger.info("📊 نتائج اكتشاف الموديلات (الأولويات المحدثة):")
        
        for provider_name, config in self.providers.items():
            if not config.enabled:
                continue
            
            logger.info(f"\n🔹 {provider_name.value.upper()}:")
            for service_type, models in config.discovered_models.items():
                if models:
                    logger.info(f"  {service_type.value} ({len(models)} models):")
                    for i, model in enumerate(models[:5]):
                        star = "⭐" if i == 0 else "  "
                        logger.info(f"    {star} {model.name} (Priority: {model.priority})")
        
        logger.info("=" * 60)
    
    def _select_best_models(self):
        """اختيار أفضل موديل لكل خدمة"""
        for provider_name, config in self.providers.items():
            if not config.enabled:
                continue
            for service_type, models in config.discovered_models.items():
                if models:
                    best_model = min(models, key=lambda x: x.priority)
                    config.active_models[service_type] = best_model.name
    
    # ==================== دوال Rate Limiting ====================
    
    async def check_rate_limit(self, user_id: int, service_type: str) -> Tuple[bool, str]:
        return await self.rate_limiter.check_rate_limit(user_id, service_type)
    
    async def get_rate_limit_stats(self, user_id: int) -> Dict[str, Any]:
        return await self.rate_limiter.get_user_stats(user_id)
    
    # ==================== دوال المزودين المتاحين ====================
    
    def get_available_providers(self, service_type: ServiceType) -> List[ProviderConfig]:
        available = []
        for provider in self.providers.values():
            if not provider.enabled or provider.usage_today >= provider.daily_limit:
                continue
            if service_type == ServiceType.CHAT:
                if provider.name in [Provider.GOOGLE, Provider.OPENAI]:
                    available.append(provider)
            elif service_type == ServiceType.IMAGE:
                if provider.name in [Provider.GOOGLE, Provider.OPENAI, Provider.STABILITY]:
                    available.append(provider)
            elif service_type == ServiceType.VIDEO:
                if provider.name in [Provider.GOOGLE, Provider.LUMA, Provider.KLING]:
                    available.append(provider)
        available.sort(key=lambda x: (x.errors_today, x.usage_today))
        return available
    
    def get_active_model(self, provider: Provider, service_type: ServiceType) -> Optional[str]:
        config = self.providers.get(provider)
        if not config or not config.enabled:
            return None
        return config.active_models.get(service_type)
    
    def rotate_model(self, provider: Provider, service_type: ServiceType, current_model: str = None) -> Optional[str]:
        config = self.providers.get(provider)
        if not config or service_type not in config.discovered_models:
            return None
        
        models = config.discovered_models[service_type]
        if not models:
            return None
        
        if not current_model or current_model not in [m.name for m in models]:
            new_model = models[0].name
        else:
            current_index = None
            for i, model in enumerate(models):
                if model.name == current_model:
                    current_index = i
                    break
            if current_index is None or current_index >= len(models) - 1:
                new_model = models[0].name
            else:
                new_model = models[current_index + 1].name
        
        config.active_models[service_type] = new_model
        logger.info(f"🔄 تدوير: {provider.value}/{service_type.value}: {current_model or 'None'} → {new_model}")
        return new_model
    
    async def _execute_with_fallback(self, provider: Provider, service_type: ServiceType,
                                   execute_func, max_retries: int = 16):
        """تنفيذ مع نظام fallback - 16 محاولة"""
        config = self.providers.get(provider)
        if not config or not config.enabled:
            raise Exception(f"المزود {provider.value} غير مفعل")
        
        current_model = self.get_active_model(provider, service_type)
        
        for attempt in range(max_retries):
            try:
                if not current_model:
                    models = config.discovered_models.get(service_type, [])
                    if not models:
                        raise Exception(f"لا توجد موديلات لـ {service_type.value}")
                    current_model = models[0].name
                    config.active_models[service_type] = current_model
                
                logger.info(f"🔄 محاولة {attempt+1}/{max_retries} مع {provider.value}/{current_model}")
                result = await execute_func(current_model)
                return result
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ خطأ في {provider.value}/{current_model}: {error_msg[:100]}")
                
                is_quota_error = any(k in error_msg.lower() for k in ['429', 'quota', 'rate limit', 'resource exhausted'])
                is_model_error = any(k in error_msg.lower() for k in ['404', 'not found', 'invalid model', 'model not found'])
                
                if is_quota_error or is_model_error:
                    next_model = self.rotate_model(provider, service_type, current_model)
                    if next_model and next_model != current_model:
                        current_model = next_model
                        continue
                    else:
                        break
                else:
                    break
        
        raise Exception(f"فشلت جميع محاولات {provider.value} ({max_retries} محاولات)")
    
    # ==================== خدمة المحادثة ====================
    
    async def chat_with_ai(self, user_id: int, message: str) -> str:
        try:
            await self.ensure_discovery()
            
            allowed, rate_message = await self.check_rate_limit(user_id, "ai_chat")
            if not allowed:
                return rate_message
            
            if not message or len(message.strip()) < 1:
                return "❌ الرسالة فارغة."
            if len(message) > 4000:
                message = message[:4000] + "..."
            
            allowed, remaining = self.check_user_limit(user_id, "ai_chat")
            if not allowed:
                return f"❌ استهلكت رصيدك اليومي من الرسائل."
            
            providers = self.get_available_providers(ServiceType.CHAT)
            if not providers:
                return "⚠️ خدمات المحادثة غير متاحة حالياً."
            
            errors = []
            for provider_config in providers:
                try:
                    provider = provider_config.name
                    async def execute_chat(model_name: str):
                        if provider == Provider.GOOGLE:
                            return await self._chat_with_google(model_name, user_id, message)
                        elif provider == Provider.OPENAI:
                            return await self._chat_with_openai(model_name, message)
                        else:
                            raise Exception(f"مزود غير مدعوم: {provider}")
                    
                    response = await self._execute_with_fallback(provider, ServiceType.CHAT, execute_chat, max_retries=16)
                    if response:
                        self.update_user_usage(user_id, "ai_chat")
                        provider_config.usage_today += 1
                        self.db.save_ai_conversation(user_id, "chat", message, response)
                        return response
                except Exception as e:
                    errors.append(f"{provider_config.name.value}: {str(e)[:100]}")
                    provider_config.errors_today += 1
                    provider_config.last_error = str(e)
                    continue
            
            return f"⚠️ جميع خدمات المحادثة فشلت:\n" + "\n".join(errors[:3])
            
        except Exception as e:
            logger.error(f"❌ خطأ في المحادثة: {e}")
            return "⚠️ حدث خطأ غير متوقع."
    
    async def _chat_with_google(self, model_name: str, user_id: int, message: str) -> str:
        try:
            self._cleanup_old_sessions()
            model = genai.GenerativeModel(model_name)
            
            if user_id not in self.chat_sessions:
                chat = model.start_chat(history=[
                    {"role": "user", "parts": ["أنت مساعد ذكي بالعربية. رد باختصار ووضوح."]},
                    {"role": "model", "parts": ["حسناً، أنا جاهز للمساعدة."]}
                ])
                self.chat_sessions[user_id] = {"chat": chat, "last_activity": datetime.now()}
            
            session = self.chat_sessions[user_id]
            session["last_activity"] = datetime.now()
            response = await asyncio.wait_for(session["chat"].send_message_async(message), timeout=30.0)
            return self._clean_response(response.text) if response and response.text else "عذراً، لم أستطع تكوين رد."
        except asyncio.TimeoutError:
            raise Exception("انتهى وقت الانتظار من Google")
        except Exception as e:
            raise Exception(f"Google Gemini error: {str(e)}")
    
    async def _chat_with_openai(self, model_name: str, message: str) -> str:
        """الدردشة مع OpenAI - يدعم GPT-5.0, GPT-5.2, GPT-4.5, GPT-4.1"""
        try:
            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.openai_client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": message}],
                        max_tokens=1000,
                        temperature=0.7
                    )
                ),
                timeout=30.0
            )
            return response.choices[0].message.content
        except asyncio.TimeoutError:
            raise Exception("انتهى وقت الانتظار من OpenAI")
        except Exception as e:
            raise Exception(f"OpenAI error: {str(e)}")
    
    def _cleanup_old_sessions(self):
        now = datetime.now()
        to_delete = [uid for uid, data in self.chat_sessions.items() if now - data["last_activity"] > self.session_timeout]
        for uid in to_delete:
            del self.chat_sessions[uid]
    
    # ==================== خدمة الصور ====================
    
    async def generate_image(self, user_id: int, prompt: str, style: str = "realistic") -> Tuple[Optional[str], str]:
        try:
            await self.ensure_discovery()
            
            allowed, rate_message = await self.check_rate_limit(user_id, "image_gen")
            if not allowed:
                return None, rate_message
            
            if not prompt or len(prompt.strip()) < 3:
                return None, "❌ الوصف قصير جداً."
            if len(prompt) > 2000:
                prompt = prompt[:2000]
            
            allowed, remaining = self.check_user_limit(user_id, "image_gen")
            if not allowed:
                return None, "❌ انتهى رصيد الصور اليومي."
            
            providers = self.get_available_providers(ServiceType.IMAGE)
            if not providers:
                return None, "⚠️ خدمات الصور غير متاحة."
            
            enhanced_prompt = await self._enhance_image_prompt(prompt, style)
            errors = []
            
            for provider_config in providers:
                try:
                    provider = provider_config.name
                    async def execute_image(model_name: str):
                        if provider == Provider.GOOGLE:
                            return await self._generate_image_google_vertex(model_name, enhanced_prompt)
                        elif provider == Provider.OPENAI:
                            return await self._generate_image_openai(model_name, enhanced_prompt)
                        elif provider == Provider.STABILITY:
                            return await self._generate_image_stability(enhanced_prompt, style, model_name)
                        else:
                            raise Exception(f"مزود غير مدعوم: {provider}")
                    
                    image_url = await self._execute_with_fallback(provider, ServiceType.IMAGE, execute_image, max_retries=16)
                    if image_url:
                        self.update_user_usage(user_id, "image_gen")
                        provider_config.usage_today += 1
                        self.db.save_generated_file(user_id, "image", prompt, image_url)
                        return image_url, "✅ تم إنشاء الصورة بنجاح"
                except Exception as e:
                    errors.append(f"{provider_config.name.value}: {str(e)[:100]}")
                    provider_config.errors_today += 1
                    provider_config.last_error = str(e)
                    continue
            
            return None, "❌ جميع خدمات الصور فشلت:\n" + "\n".join(errors[:3])
            
        except Exception as e:
            logger.error(f"❌ خطأ في توليد الصور: {e}")
            return None, "⚠️ حدث خطأ غير متوقع."
    
    async def _enhance_image_prompt(self, prompt: str, style: str) -> str:
        style_map = {
            "realistic": "photorealistic, highly detailed, natural lighting, 8k",
            "anime": "anime style, vibrant colors, large eyes, cel-shaded",
            "fantasy": "fantasy art, magical, ethereal, dramatic lighting",
            "cyberpunk": "cyberpunk, neon lights, futuristic city, high tech",
            "watercolor": "watercolor painting, artistic, soft colors, fluid"
        }
        style_desc = style_map.get(style, "photorealistic, professional photography")
        
        enhancement_prompt = f"""
        Enhance this image description for AI image generation:
        Description: {prompt}
        Style: {style} - {style_desc}
        Output: Only the enhanced description, max 300 characters.
        """
        
        try:
            google_config = self.providers[Provider.GOOGLE]
            if google_config.enabled and google_config.active_models.get(ServiceType.CHAT):
                model_name = google_config.active_models[ServiceType.CHAT]
                model = genai.GenerativeModel(model_name)
                response = await asyncio.wait_for(model.generate_content_async(enhancement_prompt), timeout=10.0)
                if response and response.text:
                    enhanced = response.text.strip()
                    if len(enhanced) > 30:
                        return enhanced[:500]
        except Exception:
            pass
        
        return f"{prompt}, {style} style, professional photography, detailed, 4k, high quality"
    
    # ========== Google Imagen ==========
    
    async def _generate_image_google_vertex(self, model_name: str, prompt: str) -> str:
        try:
            logger.info(f"🎨 Google Imagen: {model_name}...")
            
            if not self.google_project_id:
                raise Exception("❌ GOOGLE_PROJECT_ID غير موجود")
            
            if not os.path.exists("downloads"):
                os.makedirs("downloads")
            
            filename = f"downloads/imagen_{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
            api_key = self.providers[Provider.GOOGLE].api_key
            location = self.google_location
            project_id = self.google_project_id
            
            url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_name}:predict"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "instances": [{"prompt": prompt}],
                "parameters": {"sampleCount": 1, "aspectRatio": "1:1"}
            }
            
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Google Imagen Error {response.status}")
                    
                    result = await response.json()
                    predictions = result.get('predictions', [])
                    if not predictions:
                        raise Exception("❌ لا توجد تنبؤات")
                    
                    b64_data = predictions[0].get('bytesBase64Encoded') or predictions[0].get('image', {}).get('bytesBase64Encoded')
                    if not b64_data:
                        raise Exception("❌ تنسيق الصورة غير معروف")
                    
                    image_data = base64.b64decode(b64_data)
                    with open(filename, "wb") as f:
                        f.write(image_data)
                    
                    return filename
                    
        except Exception as e:
            logger.error(f"❌ Google Imagen Error: {str(e)}")
            raise e
    
    async def _generate_image_openai(self, model_name: str, prompt: str) -> str:
        """توليد صورة باستخدام OpenAI - يدعم GPT-Image-1.5 و DALL-E"""
        try:
            # GPT-Image-1.5 له endpoint مختلف
            if 'gpt-image-1.5' in model_name.lower():
                # هذا موديل جديد - استخدم نفس endpoint DALL-E مؤقتاً
                # في المستقبل هيبقى له API خاص
                response = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.openai_client.images.generate(
                            model="dall-e-3",  # مؤقتاً نستخدم DALL-E 3
                            prompt=prompt[:1000],
                            size="1024x1024",
                            quality="hd",
                            n=1
                        )
                    ),
                    timeout=60.0
                )
            else:
                response = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.openai_client.images.generate(
                            model=model_name,
                            prompt=prompt[:1000],
                            size="1024x1024",
                            quality="standard",
                            n=1
                        )
                    ),
                    timeout=60.0
                )
            return response.data[0].url
        except asyncio.TimeoutError:
            raise Exception("انتهى وقت الانتظار لـ OpenAI")
        except Exception as e:
            raise Exception(f"OpenAI Image error: {str(e)}")
    
    # ========== Stability AI ==========
    
    async def _generate_image_stability(self, prompt: str, style: str, model_name: str) -> str:
        """توليد صورة باستخدام Stability AI"""
        try:
            style_presets = {
                "realistic": "photographic",
                "anime": "anime",
                "fantasy": "fantasy-art",
                "cyberpunk": "neon-punk",
                "watercolor": None
            }
            
            if "stable-diffusion-3.5" in model_name:
                url = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
                headers = {
                    "authorization": f"Bearer {self.providers[Provider.STABILITY].api_key}",
                    "accept": "application/json"
                }
                data = {
                    "prompt": prompt,
                    "model": model_name,
                    "aspect_ratio": "1:1",
                    "seed": int(time.time()) % 1000000,
                    "output_format": "png"
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, data=data) as response:
                        if response.status == 200:
                            result = await response.json()
                            b64_data = result.get("image")
                            if b64_data:
                                if not os.path.exists("downloads"):
                                    os.makedirs("downloads")
                                filename = f"downloads/stability_sd35_{int(time.time())}.png"
                                image_data = base64.b64decode(b64_data)
                                with open(filename, "wb") as f:
                                    f.write(image_data)
                                return filename
                        raise Exception(f"Stability SD3.5 Error: {response.status}")
            else:
                base_url = "https://api.stability.ai/v1/generation"
                url = f"{base_url}/{model_name}/text-to-image"
                headers = {
                    "Authorization": f"Bearer {self.providers[Provider.STABILITY].api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                data = {
                    "text_prompts": [{"text": prompt, "weight": 1}],
                    "cfg_scale": 7,
                    "height": 1024,
                    "width": 1024,
                    "samples": 1,
                    "steps": 30,
                }
                
                style_preset = style_presets.get(style)
                if style_preset:
                    data["style_preset"] = style_preset
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=data) as response:
                        if response.status == 200:
                            result = await response.json()
                            if "artifacts" in result and len(result["artifacts"]) > 0:
                                b64_data = result["artifacts"][0]["base64"]
                                if not os.path.exists("downloads"):
                                    os.makedirs("downloads")
                                filename = f"downloads/stability_sdxl_{int(time.time())}.png"
                                image_data = base64.b64decode(b64_data)
                                with open(filename, "wb") as f:
                                    f.write(image_data)
                                return filename
                        raise Exception(f"Stability SDXL Error: {response.status}")
                        
        except Exception as e:
            logger.error(f"❌ Stability Error: {str(e)}")
            raise e
    
    # ==================== خدمة الفيديو (مع Veo 3.1 مفعل) ====================
    
    async def generate_video(self, user_id: int, prompt: str, image_url: str = None) -> Tuple[Optional[str], str]:
        """توليد فيديو - يدعم Veo 3.1, Veo 3.1 Fast, Luma, Kling"""
        try:
            await self.ensure_discovery()
            
            allowed, rate_message = await self.check_rate_limit(user_id, "video_gen")
            if not allowed:
                return None, rate_message
            
            if not prompt or len(prompt.strip()) < 5:
                return None, "❌ الوصف قصير جداً."
            if len(prompt) > 1000:
                prompt = prompt[:1000]
            
            allowed, remaining = self.check_user_limit(user_id, "video_gen")
            if not allowed:
                return None, "❌ انتهى رصيد الفيديوهات اليومي."
            
            providers = self.get_available_providers(ServiceType.VIDEO)
            if not providers:
                return None, "⚠️ خدمات الفيديو غير متاحة."
            
            enhanced_prompt = await self._enhance_video_prompt(prompt)
            errors = []
            
            for provider_config in providers:
                try:
                    provider = provider_config.name
                    async def execute_video(model_name: str):
                        if provider == Provider.GOOGLE:
                            # ========== Google Veo 3.1 ==========
                            return await self._generate_video_veo(model_name, enhanced_prompt, image_url)
                        elif provider == Provider.LUMA:
                            return await self._generate_video_luma(enhanced_prompt, image_url)
                        elif provider == Provider.KLING:
                            return await self._generate_video_kling(enhanced_prompt, image_url)
                        else:
                            raise Exception(f"مزود غير مدعوم: {provider}")
                    
                    video_url = await self._execute_with_fallback(provider, ServiceType.VIDEO, execute_video, max_retries=16)
                    if video_url:
                        self.update_user_usage(user_id, "video_gen")
                        provider_config.usage_today += 1
                        self.db.save_generated_file(user_id, "video", prompt, video_url)
                        return video_url, "✅ تم إنشاء الفيديو بنجاح"
                except Exception as e:
                    errors.append(f"{provider_config.name.value}: {str(e)[:100]}")
                    provider_config.errors_today += 1
                    provider_config.last_error = str(e)
                    continue
            
            return None, "❌ جميع خدمات الفيديو فشلت:\n" + "\n".join(errors[:3])
            
        except Exception as e:
            logger.error(f"❌ خطأ في توليد الفيديو: {e}")
            return None, "⚠️ حدث خطأ غير متوقع."
    
    async def _enhance_video_prompt(self, prompt: str) -> str:
        """تحسين وصف الفيديو"""
        enhancement_prompt = f"""
        Enhance this video description for AI video generation (Veo/Luma):
        Description: {prompt}
        Requirements: cinematic, 5 seconds, smooth camera movement, professional lighting
        Output: Only the enhanced description, max 200 characters.
        """
        try:
            google_config = self.providers[Provider.GOOGLE]
            if google_config.enabled and google_config.active_models.get(ServiceType.CHAT):
                model_name = google_config.active_models[ServiceType.CHAT]
                model = genai.GenerativeModel(model_name)
                response = await asyncio.wait_for(model.generate_content_async(enhancement_prompt), timeout=10.0)
                if response and response.text:
                    enhanced = response.text.strip()
                    if len(enhanced) > 50:
                        return enhanced[:500]
        except Exception:
            pass
        return f"{prompt}, cinematic, 5 seconds, smooth camera movement, professional lighting, 4k"
    
    # ========== Google Veo 3.1 - مفعل بالكامل ==========
    
    async def _generate_video_veo(self, model_name: str, prompt: str, image_url: str = None) -> str:
        """
        توليد فيديو باستخدام Google Veo 3.1
        يدعم: veo-3.1-generate-001, veo-3.1-fast-generate-001
        """
        try:
            logger.info(f"🎬 Google Veo: {model_name}...")
            
            if not self.veo_project_id or not self.veo_api_key:
                # إذا لم يكن Veo متاحاً، استخدم Luma كـ Fallback
                logger.warning("⚠️ Veo غير متاح، استخدام Luma كبديل")
                luma_config = self.providers[Provider.LUMA]
                if luma_config.enabled:
                    return await self._generate_video_luma(prompt, image_url)
                raise Exception("Veo غير متاح ولا يوجد Luma كبديل")
            
            if not os.path.exists("downloads"):
                os.makedirs("downloads")
            
            # Vertex AI endpoint لـ Veo
            url = f"https://{self.veo_location}-aiplatform.googleapis.com/v1/projects/{self.veo_project_id}/locations/{self.veo_location}/publishers/google/models/{model_name}:predict"
            
            headers = {
                "Authorization": f"Bearer {self.veo_api_key}",
                "Content-Type": "application/json"
            }
            
            # تحضير payload
            payload = {
                "instances": [
                    {
                        "prompt": prompt
                    }
                ],
                "parameters": {
                    "sampleCount": 1,
                    "durationSeconds": 5,  # 5 ثواني
                    "aspectRatio": "16:9",
                    "fps": 24
                }
            }
            
            # إذا كان هناك صورة مصدر
            if image_url:
                payload["instances"][0]["image"] = {"bytesBase64Encoded": image_url}
            
            # Veo 3.1 Fast له timeout أقل
            timeout_seconds = 30 if "fast" in model_name.lower() else 120
            timeout = aiohttp.ClientTimeout(total=timeout_seconds)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # بدء التوليد
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"❌ Veo Error {response.status}: {error_text[:200]}")
                        
                        # جرب Luma كـ Fallback
                        luma_config = self.providers[Provider.LUMA]
                        if luma_config.enabled:
                            logger.info("🔄 Veo فشل، استخدام Luma...")
                            return await self._generate_video_luma(prompt, image_url)
                        raise Exception(f"Veo API error: {response.status}")
                    
                    result = await response.json()
                    
                    # استخراج الفيديو من الاستجابة
                    predictions = result.get('predictions', [])
                    if not predictions:
                        raise Exception("لا توجد تنبؤات من Veo")
                    
                    # Veo يرجع فيديو encoded كـ base64
                    video_b64 = predictions[0].get('bytesBase64Encoded') or predictions[0].get('video', {}).get('bytesBase64Encoded')
                    
                    if video_b64:
                        filename = f"downloads/veo_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
                        video_data = base64.b64decode(video_b64)
                        with open(filename, "wb") as f:
                            f.write(video_data)
                        
                        logger.info(f"✅ Veo فيديو: {filename}")
                        return filename
                    else:
                        # إذا لم يرجع فيديو فوراً، قد يكون processing
                        generation_id = predictions[0].get('id') or result.get('id')
                        if generation_id:
                            # انتظر وافحص الحالة
                            for attempt in range(10):
                                await asyncio.sleep(5)
                                check_url = f"{url}/{generation_id}"
                                async with session.get(check_url, headers=headers) as check_response:
                                    if check_response.status == 200:
                                        check_data = await check_response.json()
                                        state = check_data.get('state')
                                        if state == 'SUCCEEDED':
                                            video_url = check_data.get('videoUrl') or check_data.get('generatedVideo', {}).get('videoUrl')
                                            if video_url:
                                                return video_url
                                        elif state == 'FAILED':
                                            raise Exception(f"Veo فشل: {check_data.get('error', 'غير معروف')}")
                        
                        raise Exception("لم يتم استلام فيديو من Veo")
                        
        except asyncio.TimeoutError:
            logger.error("❌ Veo: انتهى وقت الانتظار")
            # جرب Luma كـ Fallback
            luma_config = self.providers[Provider.LUMA]
            if luma_config.enabled:
                return await self._generate_video_luma(prompt, image_url)
            raise Exception("Google Veo timeout")
        except Exception as e:
            logger.error(f"❌ Veo Error: {str(e)}")
            # جرب Luma كـ Fallback
            luma_config = self.providers[Provider.LUMA]
            if luma_config.enabled:
                return await self._generate_video_luma(prompt, image_url)
            raise e
    
    async def _generate_video_luma(self, prompt: str, image_url: str = None) -> str:
        """توليد فيديو باستخدام Luma AI"""
        try:
            url = "https://api.lumalabs.ai/dream-machine/v1/generations"
            payload = {"prompt": prompt, "aspect_ratio": "16:9"}
            if image_url:
                url = "https://api.lumalabs.ai/dream-machine/v1/generations/image"
                payload["image_url"] = image_url
            
            timeout = aiohttp.ClientTimeout(total=300)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=self.luma_headers, json=payload) as response:
                    if response.status in [200, 201]:
                        data = await response.json()
                        generation_id = data.get("id")
                        if not generation_id:
                            raise Exception("لم يتم استلام معرف التوليد")
                        
                        for attempt in range(5):
                            await asyncio.sleep(10)
                            async with session.get(f"{url}/{generation_id}", headers=self.luma_headers) as check_response:
                                if check_response.status == 200:
                                    status_data = await check_response.json()
                                    state = status_data.get("state")
                                    if state == "completed":
                                        video_url = status_data.get("assets", {}).get("video")
                                        if video_url:
                                            return video_url
                                    elif state == "failed":
                                        raise Exception(f"فشل التوليد: {status_data.get('failure_reason', 'غير معروف')}")
                        raise Exception("انتهى وقت الانتظار")
                    else:
                        raise Exception(f"Luma API error: {response.status}")
        except Exception as e:
            raise Exception(f"Luma AI error: {str(e)}")
    
    async def _generate_video_kling(self, prompt: str, image_url: str = None) -> str:
        """توليد فيديو باستخدام Kling AI"""
        # TODO: تنفيذ Kling API
        luma_config = self.providers[Provider.LUMA]
        if luma_config.enabled:
            return await self._generate_video_luma(prompt, image_url)
        raise Exception("Kling AI غير متوفر حالياً")
    
    # ==================== دوال مساعدة ====================
    
    def _clean_response(self, text: str) -> str:
        if not text:
            return "عذراً، لم أستطع تكوين رد مناسب."
        try:
            clean_text = re.sub(r'THOUGHT:.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL | re.IGNORECASE)
            clean_text = clean_text.replace("THOUGHT:", "").strip()
            return clean_text[:2000] if clean_text else text[:500]
        except:
            return text[:500]
    
    def check_user_limit(self, user_id: int, service_type: str) -> Tuple[bool, int]:
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            cache_key = f"{user_id}_{today}_{service_type}"
            
            if cache_key in self.user_limits_cache:
                current_usage = self.user_limits_cache[cache_key]
                self.user_limits_cache.move_to_end(cache_key)
            else:
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        'SELECT usage_count FROM ai_usage WHERE user_id = ? AND service_type = ? AND usage_date = ?',
                        (user_id, service_type, today)
                    )
                    result = cursor.fetchone()
                    current_usage = result[0] if result else 0
                    self.user_limits_cache[cache_key] = current_usage
            
            if len(self.user_limits_cache) > self.max_cache_size:
                self.user_limits_cache.popitem(last=False)
            
            limits_config = {
                "ai_chat": int(os.getenv("DAILY_AI_LIMIT", "20")),
                "image_gen": int(os.getenv("DAILY_IMAGE_LIMIT", "5")),
                "video_gen": int(os.getenv("DAILY_VIDEO_LIMIT", "2"))
            }
            
            limit = limits_config.get(service_type, 20)
            remaining = max(0, limit - current_usage)
            return current_usage < limit, remaining
            
        except Exception as e:
            logger.error(f"❌ خطأ في فحص الحدود: {e}")
            return True, 999
    
    def update_user_usage(self, user_id: int, service_type: str) -> bool:
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            cache_key = f"{user_id}_{today}_{service_type}"
            self.user_limits_cache[cache_key] = self.user_limits_cache.get(cache_key, 0) + 1
            
            with self.db.get_connection() as conn:
                conn.execute('''
                INSERT INTO ai_usage (user_id, service_type, usage_date, usage_count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(user_id, service_type, usage_date) 
                DO UPDATE SET usage_count = usage_count + 1
                ''', (user_id, service_type, today))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"❌ خطأ في تحديث الاستخدام: {e}")
            return False
    
    def get_available_services(self) -> Dict[str, bool]:
        return {
            "chat": len(self.get_available_providers(ServiceType.CHAT)) > 0,
            "image_generation": len(self.get_available_providers(ServiceType.IMAGE)) > 0,
            "video_generation": len(self.get_available_providers(ServiceType.VIDEO)) > 0
        }
    
    def get_user_stats(self, user_id: int) -> Dict[str, int]:
        stats = {}
        today = datetime.now().strftime('%Y-%m-%d')
        for service_type in ["ai_chat", "image_gen", "video_gen"]:
            cache_key = f"{user_id}_{today}_{service_type}"
            stats[service_type] = self.user_limits_cache.get(cache_key, 0)
        return stats
    
    async def get_user_activity_stats(self, user_id: int) -> Dict[str, Any]:
        try:
            rate_stats = await self.get_rate_limit_stats(user_id)
            daily_stats = self.get_user_stats(user_id)
            
            total_usage = {"ai_chat": 0, "image_gen": 0, "video_gen": 0}
            try:
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                    SELECT service_type, SUM(usage_count) as total
                    FROM ai_usage 
                    WHERE user_id = ?
                    GROUP BY service_type
                    ''', (user_id,))
                    for row in cursor.fetchall():
                        total_usage[row[0]] = row[1] or 0
            except Exception:
                pass
            
            return {
                "rate_limiting": rate_stats,
                "daily_usage": daily_stats,
                "total_usage": total_usage,
                "limits": {
                    "ai_chat": int(os.getenv("DAILY_AI_LIMIT", "20")),
                    "image_gen": int(os.getenv("DAILY_IMAGE_LIMIT", "5")),
                    "video_gen": int(os.getenv("DAILY_VIDEO_LIMIT", "2"))
                }
            }
        except Exception as e:
            logger.error(f"❌ خطأ في get_user_activity_stats: {e}")
            return {}
    
    def get_system_stats(self) -> Dict[str, Any]:
        stats = {
            "providers": {},
            "total_requests_today": 0,
            "total_errors_today": 0,
            "discovery_completed": self.discovery_completed,
            "active_sessions": len(self.chat_sessions),
            "cache_size": len(self.user_limits_cache),
            "timestamp": datetime.now().isoformat()
        }
        
        for provider_name, config in self.providers.items():
            if config.enabled:
                stats["providers"][provider_name.value] = {
                    "enabled": True,
                    "usage_today": config.usage_today,
                    "errors_today": config.errors_today,
                    "daily_limit": config.daily_limit,
                    "remaining_limit": config.daily_limit - config.usage_today,
                    "last_error": config.last_error[:100] if config.last_error else None,
                    "active_models": config.active_models,
                    "discovered_models_count": {st.value: len(models) for st, models in config.discovered_models.items()}
                }
                stats["total_requests_today"] += config.usage_today
                stats["total_errors_today"] += config.errors_today
        
        return stats
    
    def reset_daily_counts(self):
        today = datetime.now().strftime('%Y-%m-%d')
        keys_to_delete = [k for k in self.user_limits_cache.keys() if not k.endswith(today)]
        for k in keys_to_delete:
            del self.user_limits_cache[k]
        
        for provider in self.providers.values():
            provider.usage_today = 0
            provider.errors_today = 0
            provider.last_error = None
        
        logger.info("🔄 تم إعادة تعيين العدادات اليومية")
    
    async def health_check(self) -> Dict[str, Any]:
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {},
            "rate_limiter": "active" if self.rate_limiter.cleanup_task else "starting"
        }
        
        for provider_name, config in self.providers.items():
            status["services"][provider_name.value] = {
                "enabled": config.enabled,
                "usage": config.usage_today,
                "errors": config.errors_today
            }
        
        try:
            users_count = self.db.get_users_count()
            status["database"] = f"connected ({users_count} users)"
        except Exception as e:
            status["database"] = f"error: {str(e)[:50]}"
            status["status"] = "degraded"
        
        return status


# استيراد للتوافق
AIManager = SmartAIManager
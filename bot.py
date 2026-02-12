# bot.py - النسخة المعدلة مع أمر myactivity والرسم البياني
# الإصدار: 5.4 (مع دفع الملفات + myactivity)
import os
import logging
import asyncio
import time
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from dotenv import load_dotenv
from datetime import datetime, timedelta
import io
import matplotlib
matplotlib.use('Agg')  # وضع عدم العرض (للسيرفر)
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from collections import Counter

# تحميل المتغيرات البيئية
load_dotenv()

# إعداد التسجيل
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ==================== استيراد النظام الذكي الجديد ====================
from database import db
from ai_manager import SmartAIManager as AIManager

# محاولة إضافة خط عربي للرسوم البيانية
try:
    # محاولة استخدام خط يدعم العربية
    arabic_fonts = ['Arial', 'DejaVu Sans', 'FreeSans', 'Tahoma', 'Noto Sans Arabic']
    font_found = False
    for font in arabic_fonts:
        try:
            fm.findfont(font, fallback_to_default=False)
            plt.rcParams['font.family'] = font
            font_found = True
            logger.info(f"✅ تم تحميل الخط: {font}")
            break
        except:
            continue
    if not font_found:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        logger.warning("⚠️ لم يتم العثور على خط عربي، استخدام الخط الافتراضي")
except Exception as e:
    logger.warning(f"⚠️ خطأ في تحميل الخط: {e}")

# ==================== نظام المشرفين ====================
def get_admin_ids():
    admin_ids_str = os.getenv("ADMIN_IDS", "")
    if admin_ids_str:
        try:
            return [int(admin_id.strip()) for admin_id in admin_ids_str.split(",")]
        except ValueError:
            logger.error("❌ خطأ في تنسيق ADMIN_IDS")
            return []
    return []

ADMIN_IDS = get_admin_ids()

def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS

# إنشاء كائن الذكاء الاصطناعي الذكي
ai_manager = AIManager(db)

# ==================== دوال مساعدة ====================
def check_environment():
    """فحص بيئة التشغيل"""
    logger.info("=" * 50)
    logger.info("🔍 فحص بيئة التشغيل...")
    
    required_vars = ["BOT_TOKEN", "GOOGLE_AI_API_KEY"]
    for var in required_vars:
        value = os.getenv(var)
        status = "✅ موجود" if value else "❌ مفقود"
        logger.info(f"{var}: {status}")
    
    # متغيرات إضافية للخدمات الجديدة
    optional_vars = ["GOOGLE_PROJECT_ID", "OPENAI_API_KEY", "STABILITY_API_KEY", "LUMAAI_API_KEY"]
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            logger.info(f"{var}: ✅ موجود")
    
    import sys
    logger.info(f"Python version: {sys.version}")
    logger.info("=" * 50)

# استدعاء الفحص عند البدء
check_environment()

# ==================== أوامر البوت الأساسية ====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    
    # تسجيل المستخدم في قاعدة البيانات
    db.add_or_update_user(
        user_id=user.id,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name
    )
    
    # الحصول على حالة النظام
    system_stats = ai_manager.get_system_stats()
    services = ai_manager.get_available_services()
    
    # عد المزودين النشطين
    active_providers = 0
    for p in system_stats.get("providers", {}).values():
        if p.get("enabled"):
            active_providers += 1
    
    # إرسال إشعار ترحيبي مع مميزات Veo 3.1
    await update.message.reply_text(
        f"🤖 **مرحباً {user.first_name}!**\n\n"
        f"أنا بوت الذكاء الاصطناعي الذكي المتعدد المصادر! 🚀\n\n"
        f"🎯 **ما يمكنني فعله:**\n"
        f"💬 محادثة ذكية مع {active_providers} مزود\n"
        f"🎨 إنشاء صور احترافية (SD3.5, DALL-E 3, Imagen)\n"
        f"🎬 إنشاء فيديوهات متحركة (Veo 3.1, Luma Dream Machine)\n"
        f"📊 إحصائيات استخدام ذكية مع رسوم بيانية\n\n"
        f"🔥 **المستجدات:**\n"
        f"• دعم GPT-5.0 و GPT-5.2 (أحدث موديلات OpenAI)\n"
        f"• دعم Veo 3.1 Fast (أسرع موديل فيديو من Google)\n"
        f"• نظام Rate Limiting ذكي\n\n"
        f"🔍 **معرفك:** {user.id}\n"
        f"✅ **تم التسجيل بنجاح**\n\n"
        f"📝 استخدم /help لعرض جميع الأوامر\n"
        f"📊 استخدم /myactivity لمشاهدة إحصائياتك مع رسم بياني",
        parse_mode='Markdown'
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """عرض جميع الأوامر"""
    help_text = """
🎯 **أوامر البوت الذكي (الإصدار 5.4)**

🤖 **خدمات الذكاء الاصطناعي:**
`/chat <رسالتك>` - محادثة ذكية (GPT-5.0, Gemini 2.5)
`/ask <سؤالك>` - سؤال مباشر
`/image <وصف>` - إنشاء صورة (SD3.5, DALL-E 3, Imagen)
`/draw <وصف>` - اسم بديل للصور
`/video <وصف>` - إنشاء فيديو (Veo 3.1, Luma)

📊 **معلومات النظام والاستخدام:**
`/myactivity` - إحصائياتك مع رسم بياني 📈
`/mystats` - إحصائيات استخدامك اليومي
`/limits` - حدود الاستخدام المتاحة
`/system` - حالة النظام والمزودين

👤 **الأوامر العامة:**
`/start` - بدء استخدام البوت
`/help` - عرض هذه الرسالة
`/status` - حالة البوت
`/about` - معلومات عن البوت

👑 **أوامر المشرفين:**
`/admin` - لوحة تحكم المشرفين
`/stats` - إحصائيات النظام الكاملة
`/providers` - حالة جميع المزودين

💡 **مميزات الإصدار 5.4:**
• **OpenAI:** GPT-5.0, GPT-5.2, GPT-4.5, GPT-4.1
• **Google:** Gemini 2.5, Veo 3.1 Fast, Imagen 3.0
• **Stability:** SD3.5 Large (أسرع وأجود)
• **Rate Limiting:** حماية من الإفراط في الاستخدام
• **رسوم بيانية:** إحصائيات مرئية لاستخدامك

🔧 **الدعم:** للاستفسارات تواصل مع @elbashatech
"""
    await update.message.reply_text(help_text, parse_mode='Markdown')

# ==================== أمر myactivity - مع رسم بياني ====================

async def myactivity_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    عرض إحصائيات المستخدم مع رسم بياني
    يدعم: الاستخدام اليومي، الإجمالي، Rate Limiting
    """
    user_id = update.effective_user.id
    user = update.effective_user
    first_name = user.first_name or "مستخدم"
    
    # رسالة "جاري التحميل"
    processing_msg = await update.message.reply_text(
        "📊 **جاري تحميل إحصائياتك...**\n⏳ دقيقة من فضلك",
        parse_mode='Markdown'
    )
    
    try:
        # الحصول على إحصائيات المستخدم من ai_manager
        activity_stats = await ai_manager.get_user_activity_stats(user_id)
        
        # الحصول على إحصائيات إضافية من قاعدة البيانات
        total_images = 0
        total_videos = 0
        conversations = []
        
        try:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                
                # إجمالي الصور المولدة
                cursor.execute('''
                SELECT COUNT(*) FROM ai_generated_files 
                WHERE user_id = ? AND file_type = 'image'
                ''', (user_id,))
                total_images = cursor.fetchone()[0] or 0
                
                # إجمالي الفيديوهات المولدة
                cursor.execute('''
                SELECT COUNT(*) FROM ai_generated_files 
                WHERE user_id = ? AND file_type = 'video'
                ''', (user_id,))
                total_videos = cursor.fetchone()[0] or 0
                
                # آخر 5 محادثات (للتحليل)
                cursor.execute('''
                SELECT service_type, timestamp FROM ai_conversations 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 20
                ''', (user_id,))
                conversations = cursor.fetchall()
                
        except Exception as e:
            logger.error(f"❌ خطأ في جلب إحصائيات إضافية: {e}")
        
        # ========== إعداد البيانات للرسم البياني ==========
        
        # 1. تحليل المحادثات حسب اليوم
        daily_chats = {}
        for conv in conversations:
            if conv and conv['timestamp']:
                try:
                    day = conv['timestamp'][:10]  # YYYY-MM-DD
                    daily_chats[day] = daily_chats.get(day, 0) + 1
                except:
                    pass
        
        # آخر 7 أيام
        last_7_days = []
        chat_counts = []
        
        for i in range(6, -1, -1):
            day = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            last_7_days.append(day[5:])  # MM-DD
            chat_counts.append(daily_chats.get(day, 0))
        
        # 2. بيانات الاستخدام اليومي
        daily_usage = activity_stats.get('daily_usage', {})
        limits = activity_stats.get('limits', {})
        
        # 3. بيانات Rate Limiting
        rate_stats = activity_stats.get('rate_limiting', {})
        requests_last_min = rate_stats.get('requests_last_minute', 0)
        reset_in = int(rate_stats.get('reset_in', 0))
        
        # ========== إنشاء الرسم البياني ==========
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        fig.suptitle(f'📊 إحصائيات {first_name}', fontsize=16, fontweight='bold')
        
        # الرسم البياني الأول: المحادثات اليومية
        colors1 = ['#4CAF50' if c > 0 else '#E0E0E0' for c in chat_counts]
        bars1 = ax1.bar(last_7_days, chat_counts, color=colors1, edgecolor='black', linewidth=0.5)
        ax1.set_title('المحادثات في آخر 7 أيام', fontsize=14, pad=10)
        ax1.set_xlabel('التاريخ', fontsize=12)
        ax1.set_ylabel('عدد المحادثات', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        # إضافة الأرقام على الأعمدة
        for bar, count in zip(bars1, chat_counts):
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # الرسم البياني الثاني: استخدام الخدمات اليوم
        services = ['💬 محادثات', '🎨 صور', '🎬 فيديوهات']
        usage_values = [
            daily_usage.get('ai_chat', 0),
            daily_usage.get('image_gen', 0),
            daily_usage.get('video_gen', 0)
        ]
        limit_values = [
            limits.get('ai_chat', 20),
            limits.get('image_gen', 5),
            limits.get('video_gen', 2)
        ]
        
        x_pos = np.arange(len(services))
        width = 0.35
        
        bars_usage = ax2.bar(x_pos - width/2, usage_values, width, label='المستخدم', color='#2196F3')
        bars_limit = ax2.bar(x_pos + width/2, limit_values, width, label='الحد المسموح', color='#FF9800', alpha=0.7)
        
        ax2.set_title('استخدام اليوم مقابل الحد المسموح', fontsize=14, pad=10)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(services, fontsize=11)
        ax2.set_ylabel('العدد', fontsize=12)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # إضافة الأرقام على الأعمدة
        for bar in bars_usage:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        for bar in bars_limit:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # حفظ الرسم البياني في الذاكرة
        chart_buf = io.BytesIO()
        plt.savefig(chart_buf, format='PNG', dpi=100, bbox_inches='tight')
        chart_buf.seek(0)
        plt.close()
        
        # ========== بناء رسالة الإحصائيات ==========
        
        # أشرطة التقدم للخدمات
        progress_bars = ""
        for service, used, limit in zip(['ai_chat', 'image_gen', 'video_gen'], 
                                       usage_values, 
                                       [limits.get('ai_chat', 20), limits.get('image_gen', 5), limits.get('video_gen', 2)]):
            service_name = {
                'ai_chat': '💬 المحادثات',
                'image_gen': '🎨 الصور',
                'video_gen': '🎬 الفيديوهات'
            }.get(service, service)
            
            percentage = min(100, int((used / limit) * 100)) if limit > 0 else 0
            filled = int(percentage / 10)
            bar = '🟩' * filled + '⬜' * (10 - filled)
            progress_bars += f"\n{service_name}: {bar} {used}/{limit} ({percentage}%)"
        
        # إجمالي الاستخدام
        total_usage = activity_stats.get('total_usage', {})
        total_chats = total_usage.get('ai_chat', 0)
        total_images_all = total_usage.get('image_gen', total_images)  # استخدام القيمة من DB
        total_videos_all = total_usage.get('video_gen', total_videos)
        
        # معلومات Rate Limiting
        rate_info = ""
        if requests_last_min > 0:
            rate_info = f"\n⏱️ **طلبات في آخر دقيقة:** {requests_last_min}"
            if reset_in > 0:
                rate_info += f"\n⏳ **تجديد خلال:** {reset_in} ثانية"
        
        # إحصائيات المزودين المستخدمين (من السجل)
        providers_used = []
        try:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT details FROM activity_logs 
                WHERE user_id = ? AND action = 'generated_file'
                ORDER BY timestamp DESC LIMIT 10
                ''', (user_id,))
                logs = cursor.fetchall()
                for log in logs:
                    if log and log['details']:
                        if 'provider=' in log['details']:
                            provider = log['details'].split('provider=')[1].split(',')[0]
                            providers_used.append(provider)
        except:
            pass
        
        # إحصائيات المزودين
        provider_stats = ""
        if providers_used:
            provider_counter = Counter(providers_used)
            top_providers = provider_counter.most_common(3)
            provider_stats = "\n🔧 **المزودون الأكثر استخداماً:**\n"
            for provider, count in top_providers:
                provider_stats += f"• {provider}: {count} مرة\n"
        
        # رسالة الإحصائيات النصية
        stats_text = f"""
📊 **إحصائيات {first_name}** 🆔 `{user_id}`

📅 **اليوم:** {datetime.now().strftime('%Y-%m-%d')}
{progress_bars}

📈 **إجمالي الاستخدام:**
💬 محادثات: {total_chats:,}
🎨 صور مولدة: {total_images_all:,}
🎬 فيديوهات: {total_videos_all:,}
✨ إجمالي الطلبات: {total_chats + total_images_all + total_videos_all:,}

⚡ **حالة النظام:**
• Rate Limiting: {'✅ نشط' if ai_manager.rate_limiter.cleanup_task else '⏳ قيد التشغيل'}
{rate_info}
{provider_stats}

🔄 **التجديد:** منتصف الليل (توقيت UTC)
📱 استخدم /limits لمعرفة الحدود الكاملة
"""
        
        # حذف رسالة "جاري التحميل"
        await processing_msg.delete()
        
        # إرسال الرسم البياني + الإحصائيات
        await update.message.reply_photo(
            photo=chart_buf,
            caption=stats_text,
            parse_mode='Markdown'
        )
        
        # إضافة أزرار تفاعلية
        keyboard = [
            [
                InlineKeyboardButton("🔄 تحديث", callback_data="refresh_activity"),
                InlineKeyboardButton("📊 حدودي", callback_data="my_limits")
            ],
            [
                InlineKeyboardButton("🎨 صوري", callback_data="my_images"),
                InlineKeyboardButton("🎬 فيديوهاتي", callback_data="my_videos")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "📌 **اختر من القائمة:**",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
        
    except Exception as e:
        logger.error(f"❌ خطأ في myactivity: {e}", exc_info=True)
        await processing_msg.delete()
        await update.message.reply_text(
            "❌ **حدث خطأ أثناء تحميل الإحصائيات**\n\n"
            "جرب مرة أخرى بعد قليل.\n"
            "إذا استمرت المشكلة، استخدم /mystats لعرض إحصائيات مبسطة.",
            parse_mode='Markdown'
        )

# ==================== معالج الأزرار التفاعلية (Callback) ====================

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج الضغط على الأزرار"""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    data = query.data
    
    if data == "refresh_activity":
        # إعادة تحميل الإحصائيات
        await query.edit_message_text("🔄 **جاري تحديث الإحصائيات...**")
        
        # محاكاة الأمر myactivity
        update.effective_user = query.from_user
        update.message = query.message
        await myactivity_command(update, context)
        
    elif data == "my_limits":
        # عرض الحدود
        limits_text = """
📊 **حدود الاستخدام اليومية:**

🤖 **الذكاء الاصطناعي:**
💬 المحادثات: 20 رسالة
🎨 الصور المولدة: 5 صور  
🎬 الفيديوهات: 2 فيديو

⚡ **نظام Rate Limiting:**
• محادثة: رسالة كل ثانية (20/دقيقة)
• صور: صورة كل 2 ثانية (5/دقيقة)
• فيديو: فيديو كل 10 ثواني (2/دقيقة)

🔄 **التجديد:** تلقائي كل 24 ساعة
"""
        await query.edit_message_text(limits_text, parse_mode='Markdown')
        
    elif data == "my_images":
        # عرض آخر الصور
        images = db.get_user_generated_files(user_id, 'image', limit=5)
        if images:
            text = "🖼️ **آخر صورك المولدة:**\n\n"
            for img in images[:3]:
                date = img['created_at'][:16] if img['created_at'] else 'تاريخ غير معروف'
                text += f"• `{img['prompt'][:50]}...`\n  📅 {date}\n\n"
            text += "📌 استخدم /gallery لعرض الكل (قريباً)"
            await query.edit_message_text(text, parse_mode='Markdown')
        else:
            await query.edit_message_text("📭 لم تقم بإنشاء أي صور بعد!", parse_mode='Markdown')
            
    elif data == "my_videos":
        # عرض آخر الفيديوهات
        videos = db.get_user_generated_files(user_id, 'video', limit=5)
        if videos:
            text = "🎬 **آخر فيديوهاتك المولدة:**\n\n"
            for vid in videos[:3]:
                date = vid['created_at'][:16] if vid['created_at'] else 'تاريخ غير معروف'
                text += f"• `{vid['prompt'][:50]}...`\n  📅 {date}\n\n"
            await query.edit_message_text(text, parse_mode='Markdown')
        else:
            await query.edit_message_text("📭 لم تقم بإنشاء أي فيديوهات بعد!", parse_mode='Markdown')

# ==================== أوامر الإحصائيات المبسطة ====================

async def my_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """إحصائيات استخدامي (نسخة مبسطة)"""
    user_id = update.effective_user.id
    user = update.effective_user
    username = user.first_name or "مستخدم"
    
    stats = ai_manager.get_user_stats(user_id)
    services = ai_manager.get_available_services()
    system_stats = ai_manager.get_system_stats()
    
    stats_text = f"📊 **إحصائيات {username}**\n\n"
    stats_text += f"🆔 المعرف: {user_id}\n"
    stats_text += f"📅 اليوم: {datetime.now().strftime('%Y-%m-%d')}\n\n"
    
    limits = {
        "ai_chat": int(os.getenv("DAILY_AI_LIMIT", "20")),
        "image_gen": int(os.getenv("DAILY_IMAGE_LIMIT", "5")),
        "video_gen": int(os.getenv("DAILY_VIDEO_LIMIT", "2"))
    }
    
    for service, limit in limits.items():
        used = stats.get(service, 0)
        remaining = max(0, limit - used)
        percentage = (used / limit * 100) if limit > 0 else 0
        
        service_names = {
            "ai_chat": "💬 المحادثات",
            "image_gen": "🎨 الصور المولدة",
            "video_gen": "🎬 الفيديوهات"
        }
        
        filled_blocks = int(percentage / 10)
        progress_bar = "🟩" * filled_blocks + "⬜" * (10 - filled_blocks)
        
        stats_text += f"{service_names.get(service, service)}:\n"
        stats_text += f"{progress_bar}\n"
        stats_text += f"📊 {used}/{limit} ({remaining} متبقي)\n\n"
    
    stats_text += "📈 **لرؤية إحصائيات متقدمة مع رسم بياني:**\n"
    stats_text += "➡️ استخدم `/myactivity`"
    
    await update.message.reply_text(stats_text, parse_mode='Markdown')

# ==================== أوامر الذكاء الاصطناعي ====================

async def chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """بدء محادثة مع الذكاء الاصطناعي"""
    user_id = update.effective_user.id
    user_message = ' '.join(context.args) if context.args else ""
    
    if not user_message:
        await update.message.reply_text(
            "💬 **المحادثة الذكية**\n\n"
            "اكتب رسالتك بعد الأمر:\n"
            "`/chat مرحبا، كيف حالك؟`\n\n"
            "✨ **المميزات:**\n"
            "• يستخدم GPT-5.0 أولاً (الأحدث)\n"
            "• يتبدل لـ GPT-5.2, GPT-4.5, GPT-4.1 تلقائياً\n"
            "• يحفظ سياق المحادثة",
            parse_mode='Markdown'
        )
        return
    
    processing_msg = await update.message.reply_text(
        "🤔 **جاري التفكير...**\n"
        "⚡ النظام الذكي يختار أفضل مزود"
    )
    
    start_time = time.time()
    
    try:
        response = await ai_manager.chat_with_ai(user_id, user_message)
        response_time = time.time() - start_time
        
        await update.message.reply_text(
            f"🤖 **المساعد الذكي:**\n\n{response}\n\n"
            f"⏱️ الوقت: {response_time:.1f} ثانية\n"
            f"⚡ النظام الذكي يعمل بكفاءة",
            parse_mode='Markdown'
        )
    except Exception as e:
        logger.error(f"❌ Chat command error: {e}")
        await update.message.reply_text(
            "⚠️ **حدث خطأ مؤقت**\n\n"
            "النظام يحاول مزوداً آخر...\n"
            "جرب مرة أخرى بعد قليل."
        )
    finally:
        try:
            await processing_msg.delete()
        except:
            pass

async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """إنشاء صورة باستخدام النظام الذكي"""
    user_id = update.effective_user.id
    
    if not context.args:
        await update.message.reply_text(
            "🎨 **إنشاء صور ذكية**\n\n"
            "**الاستخدام:** `/image <وصف الصورة> [النمط]`\n\n"
            "**أمثلة:**\n"
            "`/image قطة لطيفة تجلس على كرسي`\n"
            "`/image منظر لغروب الشمس realistic`\n\n"
            "**الأنماط المتاحة:**\n"
            "`realistic` - واقعي (افتراضي)\n"
            "`anime` - أنمي / كرتون\n"
            "`fantasy` - فنتازيا\n"
            "`cyberpunk` - مستقبلي\n"
            "`watercolor` - ألوان مائية\n\n"
            "⚡ **النظام الذكي:**\n"
            "• يستخدم Stable Diffusion 3.5 Large أولاً\n"
            "• يتبدل لـ DALL-E 3, Imagen تلقائياً\n"
            "• يحسن الوصف أوتوماتيكياً",
            parse_mode='Markdown'
        )
        return
    
    args = context.args
    prompt_words = args[:-1]
    style = args[-1] if args[-1] in ["realistic", "anime", "fantasy", "cyberpunk", "watercolor"] else "realistic"
    
    if style != args[-1]:
        prompt_words = args
    
    prompt = ' '.join(prompt_words)
    
    if len(prompt) < 3:
        await update.message.reply_text("❌ الرجاء إدخال وصف أطول للصورة (3 كلمات على الأقل)")
        return
    
    wait_msg = await update.message.reply_text(
        "🎨 **جاري إنشاء صورتك...**\n"
        "⚡ النظام الذكي يعمل:\n"
        "1. تحسين الوصف تلقائياً\n"
        "2. اختيار أفضل مزود (SD3.5, DALL-E 3, Imagen)\n"
        "3. التبديل الذكي إذا لزم\n"
        "⏳ قد يستغرق 10-30 ثانية"
    )
    
    try:
        start_time = time.time()
        image_url, message = await ai_manager.generate_image(user_id, prompt, style)
        response_time = time.time() - start_time
        
        if image_url:
            await update.message.reply_photo(
                photo=image_url,
                caption=f"✅ **تم إنشاء صورتك بنجاح!**\n\n"
                       f"📝 **الوصف:** {prompt}\n"
                       f"🎨 **النمط:** {style}\n"
                       f"⏱️ **الوقت:** {response_time:.1f} ثانية\n"
                       f"⚡ **المزود:** {ai_manager.get_active_model(Provider.STABILITY, ServiceType.IMAGE) or 'SD3.5'}\n\n"
                       f"💾 تم حفظ الصورة في مكتبتك\n"
                       f"📊 استخدم `/myactivity` لعرض إحصائياتك",
                parse_mode='Markdown'
            )
            
            # تسجيل النشاط
            db.log_activity(user_id, "generated_file", f"type=image,provider=stability")
            
        else:
            await update.message.reply_text(
                f"❌ **لم نتمكن من إنشاء الصورة**\n\n"
                f"{message}\n\n"
                f"✨ **الحلول المقترحة:**\n"
                f"1. حاول بوصف مختلف\n"
                f"2. استخدم نمطاً آخر\n"
                f"3. انتظر قليلاً وجرب مرة أخرى"
            )
        
        await wait_msg.delete()
        
    except Exception as e:
        logger.error(f"❌ Image command error: {e}")
        await update.message.reply_text(
            "❌ **حدث خطأ غير متوقع**\n\n"
            "النظام يحاول إصلاح نفسه تلقائياً..."
        )
        try:
            await wait_msg.delete()
        except:
            pass

async def video_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """إنشاء فيديو باستخدام النظام الذكي (يدعم Veo 3.1)"""
    user_id = update.effective_user.id
    
    if not context.args:
        await update.message.reply_text(
            "🎬 **إنشاء فيديو ذكي**\n\n"
            "**الاستخدام:**\n"
            "`/video منظر طبيعي لغروب الشمس`\n\n"
            "**أمثلة:**\n"
            "`/video مدينة المستقبل بإضاءة نيون`\n"
            "`/video بحر هائج بأمواج عالية`\n\n"
            "⚡ **المزودون النشطون:**\n"
            "• **Google Veo 3.1 Fast** - الأسرع (الأولوية)\n"
            "• **Google Veo 3.1** - جودة عالية\n"
            "• **Luma Dream Machine** - احتياطي\n\n"
            "⚠️ **المدة:** 30 ثانية - 2 دقيقة (Veo Fast)\n"
            "⚠️ **المدة:** 2-5 دقائق (Veo/Luma)",
            parse_mode='Markdown'
        )
        return
    
    prompt = ' '.join(context.args)
    
    if len(prompt) < 4:
        await update.message.reply_text("❌ الرجاء إدخال وصف أطول للفيديو (4 كلمات على الأقل)")
        return
    
    image_url = None
    if update.message.reply_to_message and update.message.reply_to_message.photo:
        photo = update.message.reply_to_message.photo[-1]
        image_file = await photo.get_file()
        image_url = image_file.file_path
    
    wait_msg = await update.message.reply_text(
        "🎬 **جاري إنشاء الفيديو...**\n"
        "⚡ النظام الذكي يعمل:\n"
        "1. تحسين الوصف سينمائياً\n"
        "2. محاولة Google Veo 3.1 Fast (الأسرع)\n"
        "3. التبديل لـ Veo 3.1 أو Luma\n"
        "⏳ قد يستغرق 30 ثانية - 5 دقائق\n"
        "📱 يمكنك متابعة استخدام البوت"
    )
    
    try:
        start_time = time.time()
        video_url, message = await ai_manager.generate_video(user_id, prompt, image_url)
        response_time = time.time() - start_time
        
        if video_url:
            # تحديد المزود المستخدم
            provider_used = "Veo 3.1 Fast"
            if "fast" not in str(video_url).lower():
                provider_used = "Veo 3.1"
            if "luma" in str(video_url).lower():
                provider_used = "Luma AI"
            
            await update.message.reply_video(
                video=video_url,
                caption=f"✅ **تم إنشاء الفيديو بنجاح!**\n\n"
                       f"📝 **الوصف:** {prompt}\n"
                       f"⏱️ **الوقت:** {response_time:.1f} ثانية\n"
                       f"⚡ **المزود:** {provider_used}\n\n"
                       f"💾 تم حفظ الفيديو في مكتبتك\n"
                       f"📊 استخدم `/myactivity` لعرض إحصائياتك",
                parse_mode='Markdown'
            )
            
            # تسجيل النشاط
            db.log_activity(user_id, "generated_file", f"type=video,provider={provider_used}")
            
        else:
            await update.message.reply_text(
                f"❌ **لم نتمكن من إنشاء الفيديو**\n\n"
                f"{message}\n\n"
                f"✨ **الحلول المقترحة:**\n"
                f"1. حاول بوصف مختلف\n"
                f"2. انتظر 5 دقائق وجرب مرة أخرى"
            )
        
        await wait_msg.delete()
        
    except Exception as e:
        logger.error(f"❌ Video command error: {e}")
        await update.message.reply_text(
            "❌ **حدث خطأ غير متوقع**\n\n"
            "خدمة الفيديو قد تكون مشغولة حالياً..."
        )
        try:
            await wait_msg.delete()
        except:
            pass

# ==================== معالج المحادثات العادية ====================

async def handle_ai_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالجة المحادثات العادية مع النظام الذكي"""
    user_id = update.effective_user.id
    user_message = update.message.text
    
    if user_message.startswith('/'):
        return
    
    is_reply_to_ai = (
        update.message.reply_to_message and 
        update.message.reply_to_message.from_user.id == context.bot.id
    )
    is_direct_chat = not update.message.reply_to_message
    
    if is_reply_to_ai or is_direct_chat:
        processing_msg = await update.message.reply_text(
            "🤔 **جاري التفكير...**\n"
            "⚡ النظام الذكي يعالج طلبك"
        )
        
        try:
            response = await ai_manager.chat_with_ai(user_id, user_message)
            reply_text = f"🤖 **المساعد الذكي:**\n\n{response}"
            
            if len(reply_text) > 4000:
                parts = [reply_text[i:i+4000] for i in range(0, len(reply_text), 4000)]
                for part in parts:
                    await update.message.reply_text(part, parse_mode='Markdown')
            else:
                await update.message.reply_text(reply_text, parse_mode='Markdown')
                
        except Exception as e:
            logger.error(f"❌ AI conversation error: {e}")
            await update.message.reply_text(
                "⚠️ **الخدمة مشغولة حالياً**\n\n"
                "النظام يحاول مزوداً آخر تلقائياً..."
            )
        finally:
            try:
                await processing_msg.delete()
            except:
                pass

# ==================== أوامر المشرفين ====================

async def admin_panel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if not is_admin(user_id):
        await update.message.reply_text("⛔ هذا الأمر للمشرفين فقط!")
        return
    
    users_count = db.get_users_count()
    system_stats = ai_manager.get_system_stats()
    
    active_providers = 0
    for p in system_stats.get("providers", {}).values():
        if p.get("enabled"):
            active_providers += 1
    
    admin_commands = f"""
👑 **لوحة تحكم المشرفين (النظام الذكي 5.4)**

🤖 **حالة النظام الذكي:**
🔧 مزودون نشطون: {active_providers}
📤 طلبات اليوم: {system_stats.get('total_requests_today', 0)}
❌ أخطاء اليوم: {system_stats.get('total_errors_today', 0)}
⚡ Rate Limiter: {'✅ نشط' if ai_manager.rate_limiter.cleanup_task else '⏳ قيد التشغيل'}

📊 **الإحصائيات:**
/stats - إحصائيات النظام الكاملة
/userslist - عرض المستخدمين ({users_count} مستخدم)
/providers - حالة جميع المزودين

📢 **الإذاعة:**
/broadcast - إعداد رسالة للإذاعة
/sendbroadcast - إرسال الرسالة المعلقة

🔧 **إدارة النظام:**
/resetcache - إعادة تعيين الكاش
/systemlogs - سجلات النظام

🔢 **معلومات النظام:**
👥 المستخدمين: {users_count}
👑 المشرفين: {len(ADMIN_IDS)}
⚡ مزودون AI: {active_providers} نشط
💾 قاعدة البيانات: ✅ نشطة
"""
    
    await update.message.reply_text(admin_commands, parse_mode='Markdown')
    logger.info(f"المشرف {user_id} فتح لوحة التحكم")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """عرض إحصائيات النظام الكاملة"""
    user_id = update.effective_user.id
    
    if not is_admin(user_id):
        await update.message.reply_text("⛔ هذا الأمر للمشرفين فقط!")
        return
    
    try:
        stats = db.get_stats_fixed()
        system_stats = ai_manager.get_system_stats()
        
        stats_text = f"""
📊 **إحصائيات النظام الكاملة (النظام الذكي 5.4)**

👥 **المستخدمون:**
👤 العدد الكلي: {stats['total_users']} مستخدم
🆕 الجدد اليوم: {stats.get('new_users_today', 0)}
💬 الرسائل الكلية: {stats.get('total_messages', 0):,}

🤖 **النظام الذكي:**
🔧 مزودون نشطون: {len([p for p in system_stats.get("providers", {}).values() if p.get("enabled")])}
📤 طلبات اليوم: {system_stats.get('total_requests_today', 0):,}
❌ أخطاء اليوم: {system_stats.get('total_errors_today', 0):,}
⚡ جلسات نشطة: {system_stats.get('active_sessions', 0)}
💾 حجم الكاش: {system_stats.get('cache_size', 0)}

📊 **إحصائيات الذكاء الاصطناعي:"""
        
        try:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(DISTINCT user_id) FROM ai_usage")
                ai_users = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT SUM(usage_count) FROM ai_usage WHERE service_type = 'ai_chat'")
                total_chats = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT SUM(usage_count) FROM ai_usage WHERE service_type = 'image_gen'")
                total_images = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT SUM(usage_count) FROM ai_usage WHERE service_type = 'video_gen'")
                total_videos = cursor.fetchone()[0] or 0
                
                stats_text += f"""
👤 مستخدمون AI: {ai_users}
💬 محادثات: {total_chats:,}
🎨 صور مولدة: {total_images:,}
🎬 فيديوهات: {total_videos:,}
"""
        except Exception as e:
            logger.error(f"❌ خطأ في إحصائيات AI: {e}")
        
        stats_text += f"""
📢 **الإذاعات:**
📤 عدد الإذاعات: {stats.get('total_broadcasts', 0)}
"""
        
        await update.message.reply_text(stats_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ خطأ في عرض الإحصائيات: {e}")
        await update.message.reply_text("📊 **حالة النظام:**\n\n✅ النظام الذكي يعمل بشكل طبيعي")

async def providers_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """عرض حالة جميع المزودين"""
    user_id = update.effective_user.id
    
    if not is_admin(user_id):
        await update.message.reply_text("⛔ هذا الأمر للمشرفين فقط!")
        return
    
    try:
        system_stats = ai_manager.get_system_stats()
        
        providers_text = "🔧 **حالة جميع المزودين:**\n\n"
        
        for provider_name, provider_info in system_stats.get("providers", {}).items():
            status = "✅" if provider_info.get("enabled") else "❌"
            usage = provider_info.get("usage_today", 0)
            limit = provider_info.get("daily_limit", 100)
            errors = provider_info.get("errors_today", 0)
            
            providers_text += f"{status} **{provider_name.upper()}:**\n"
            providers_text += f"   📊 الاستخدام: {usage}/{limit}\n"
            providers_text += f"   ❌ الأخطاء: {errors}\n"
            
            if provider_info.get("active_models"):
                providers_text += f"   🤖 الموديلات النشطة:\n"
                for service, model in provider_info.get("active_models", {}).items():
                    providers_text += f"      • {service}: {model}\n"
            
            providers_text += "\n"
        
        await update.message.reply_text(providers_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ خطأ في عرض المزودين: {e}")
        await update.message.reply_text("⚠️ حدث خطأ في جلب حالة المزودين.")

async def reset_cache_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """إعادة تعيين الكاش"""
    user_id = update.effective_user.id
    
    if not is_admin(user_id):
        await update.message.reply_text("⛔ هذا الأمر للمشرفين فقط!")
        return
    
    try:
        ai_manager.reset_daily_counts()
        ai_manager.user_limits_cache.clear()
        
        await update.message.reply_text(
            "🔄 **تم إعادة تعيين الكاش بنجاح!**\n\n"
            "✅ تمت إعادة تعيين:\n"
            "• عدادات المزودين اليومية\n"
            "• كاش حدود المستخدمين\n"
            "• سجلات الأخطاء\n\n"
            "✨ النظام جاهز ليوم جديد!"
        )
        logger.info(f"🔄 المشرف {user_id} أعاد تعيين الكاش")
        
    except Exception as e:
        logger.error(f"❌ خطأ في إعادة تعيين الكاش: {e}")
        await update.message.reply_text("❌ فشل إعادة تعيين الكاش.")

# ==================== دوال الإذاعة ====================

async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """إعداد رسالة إذاعة"""
    user_id = update.effective_user.id
    
    if not is_admin(user_id):
        await update.message.reply_text("⛔ هذا الأمر للمشرفين فقط!")
        return
    
    if update.message.reply_to_message:
        message = update.message.reply_to_message.text or "رسالة ميديا"
        users_count = db.get_users_count()
        
        await update.message.reply_text(
            f"📢 **رسالة الإذاعة:**\n"
            f"'{message[:50]}...'\n\n"
            f"👥 عدد المستهدفين: {users_count} مستخدم\n"
            f"✅ جاهزة للإرسال\n\n"
            f"ℹ️ *لإرسال فعلياً:*\n"
            f"أرسل /sendbroadcast",
            parse_mode='Markdown'
        )
        
        context.user_data['pending_broadcast'] = message
    else:
        await update.message.reply_text(
            "📝 **طريقة استخدام /broadcast:**\n"
            "1. أرسل الرسالة التي تريد إذاعتها\n"
            "2. رد على الرسالة بالأمر /broadcast\n\n"
            "✅ **المميزات:**\n"
            "- الإرسال لجميع المستخدمين\n"
            "- تتبع من استلم الرسالة",
            parse_mode='Markdown'
        )

async def send_broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """إرسال الإذاعة"""
    user_id = update.effective_user.id
    
    if not is_admin(user_id):
        await update.message.reply_text("⛔ هذا الأمر للمشرفين فقط!")
        return
    
    if 'pending_broadcast' not in context.user_data:
        await update.message.reply_text("❌ لا توجد رسالة معلقة للإذاعة!\nاستخدم /broadcast أولاً")
        return
    
    message = context.user_data['pending_broadcast']
    users = db.get_all_users()
    users_count = len(users)
    
    if users_count == 0:
        await update.message.reply_text("❌ لا يوجد مستخدمين لإرسال الإذاعة لهم!")
        return
    
    broadcast_id = db.add_broadcast(user_id, message, users_count)
    
    if not broadcast_id:
        await update.message.reply_text("❌ فشل في حفظ الإذاعة!")
        return
    
    sent_count = 0
    failed_count = 0
    
    await update.message.reply_text(
        f"📤 جاري إرسال الإذاعة لـ {users_count} مستخدم...\n"
        f"⏳ قد يستغرق بعض الوقت..."
    )
    
    for user in users:
        try:
            if user['user_id'] != user_id:
                await context.bot.send_message(
                    chat_id=user['user_id'],
                    text=f"📢 **إذاعة من الإدارة:**\n\n{message}"
                )
                sent_count += 1
            
            if sent_count % 10 == 0:
                await asyncio.sleep(0.3)
                
        except Exception as e:
            failed_count += 1
            logger.error(f"❌ فشل إرسال للإذاعة للمستخدم {user['user_id']}: {e}")
    
    success_rate = (sent_count / users_count * 100) if users_count > 0 else 0
    
    report = f"""
✅ **تم إرسال الإذاعة بنجاح!**

📊 **التقرير:**
🆔 رقم الإذاعة: {broadcast_id}
👥 العدد الكلي: {users_count} مستخدم
✅ تم الإرسال بنجاح: {sent_count}
❌ فشل الإرسال: {failed_count}
📈 نسبة النجاح: {success_rate:.1f}%
"""
    
    await update.message.reply_text(report, parse_mode='Markdown')
    del context.user_data['pending_broadcast']

async def users_list_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """قائمة المستخدمين"""
    user_id = update.effective_user.id
    
    if not is_admin(user_id):
        await update.message.reply_text("⛔ هذا الأمر للمشرفين فقط!")
        return
    
    users = db.get_all_users()
    users_count = len(users)
    
    if users_count == 0:
        await update.message.reply_text("📭 لا يوجد مستخدمين مسجلين بعد.")
        return
    
    display_users = users[:10]
    
    users_text = f"👥 **المستخدمون المسجلون** ({users_count} مستخدم)\n\n"
    
    for i, user in enumerate(display_users, 1):
        users_text += f"{i}. {user['first_name']}"
        if user['username']:
            users_text += f" (@{user['username']})"
        users_text += f" - ID: {user['user_id']}\n"
        join_date = user['join_date'][:10] if user['join_date'] else "غير معروف"
        users_text += f"   📅 انضم: {join_date}\n"
        users_text += f"   💬 رسائل: {user['message_count']}\n\n"
    
    if users_count > 10:
        users_text += f"\n📋 عرض 10 من أصل {users_count} مستخدم"
    
    await update.message.reply_text(users_text, parse_mode='Markdown')

# ==================== دوال مساعدة ====================

def check_database_status():
    """فحص حالة قاعدة البيانات"""
    try:
        users_count = db.get_users_count()
        stats = db.get_stats_fixed()
        
        status_info = {
            'database_file': db.db_name,
            'users_count': users_count,
            'stats_available': bool(stats),
            'last_check': datetime.now().isoformat()
        }
        
        logger.info(f"✅ حالة قاعدة البيانات: {status_info}")
        return status_info
        
    except Exception as e:
        logger.error(f"❌ فشل في فحص حالة قاعدة البيانات: {e}")
        return {'error': str(e), 'last_check': datetime.now().isoformat()}

def setup_handlers(application):
    """إعداد معالجات الأوامر والرسائل"""
    
    # الأوامر الأساسية
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", system_command))
    application.add_handler(CommandHandler("system", system_command))
    application.add_handler(CommandHandler("about", about_command))
    application.add_handler(CommandHandler("limits", limits_command))
    
    # أوامر الإحصائيات - مع الرسم البياني
    application.add_handler(CommandHandler("myactivity", myactivity_command))
    application.add_handler(CommandHandler("myusage", myactivity_command))  # اسم بديل
    application.add_handler(CommandHandler("mystats", my_stats_command))
    application.add_handler(CommandHandler("aistats", my_stats_command))
    
    # أوامر الذكاء الاصطناعي
    application.add_handler(CommandHandler("chat", chat_command))
    application.add_handler(CommandHandler("ask", chat_command))
    application.add_handler(CommandHandler("image", image_command))
    application.add_handler(CommandHandler("draw", image_command))
    application.add_handler(CommandHandler("video", video_command))
    application.add_handler(CommandHandler("aihelp", help_command))
    
    # أوامر المشرفين
    application.add_handler(CommandHandler("admin", admin_panel))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("providers", providers_command))
    application.add_handler(CommandHandler("resetcache", reset_cache_command))
    application.add_handler(CommandHandler("broadcast", broadcast_command))
    application.add_handler(CommandHandler("sendbroadcast", send_broadcast_command))
    application.add_handler(CommandHandler("userslist", users_list_command))
    
    # معالج الأزرار التفاعلية
    application.add_handler(CallbackQueryHandler(button_callback_handler))
    
    # معالج المحادثات العادية مع AI
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        handle_ai_conversation
    ), group=1)
    
    # معالج للردود على الإذاعات
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        handle_broadcast_reply
    ), group=2)

async def system_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """عرض حالة النظام الذكي والمزودين"""
    try:
        system_stats = ai_manager.get_system_stats()
        services = ai_manager.get_available_services()
        
        status_text = "⚙️ **حالة النظام الذكي المتعدد المصادر**\n\n"
        
        status_text += "📊 **الخدمات المتاحة:**\n"
        status_text += f"💬 المحادثة: {'✅ متاحة' if services.get('chat') else '❌ غير متاحة'}\n"
        status_text += f"🎨 إنشاء الصور: {'✅ متاحة' if services.get('image_generation') else '❌ غير متاحة'}\n"
        status_text += f"🎬 إنشاء الفيديوهات: {'✅ متاحة' if services.get('video_generation') else '❌ غير متاحة'}\n\n"
        
        active_providers = 0
        providers_text = "🔧 **المزودون النشطون:**\n"
        
        for provider_name, provider_info in system_stats.get("providers", {}).items():
            if provider_info.get("enabled"):
                active_providers += 1
                providers_text += f"• {provider_name.upper()}: {provider_info.get('usage_today', 0)} طلب\n"
        
        status_text += providers_text + "\n"
        
        status_text += f"📈 **إحصائيات اليوم:**\n"
        status_text += f"📤 الطلبات: {system_stats.get('total_requests_today', 0)}\n"
        status_text += f"❌ الأخطاء: {system_stats.get('total_errors_today', 0)}\n"
        status_text += f"🔄 المزودون: {active_providers}/{len(system_stats.get('providers', {}))}\n"
        status_text += f"⚡ Rate Limiter: {'✅ نشط' if ai_manager.rate_limiter.cleanup_task else '⏳ قيد التشغيل'}\n\n"
        
        status_text += "✨ **النظام يعمل بشكل ذكي ومستقر**"
        
        await update.message.reply_text(status_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ خطأ في أمر النظام: {e}")
        await update.message.reply_text("✅ النظام يعمل، لكن هناك تأخير في جلب التفاصيل.")

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """عرض معلومات عن البوت"""
    about_text = """
🤖 **معلومات البوت الذكي**

الإصدار: 5.4 (النظام الذكي المتعدد المصادر)
التاريخ: 2026

🎯 **المميزات الرئيسية:**
1. دعم GPT-5.0 / GPT-5.2 (أحدث موديلات OpenAI)
2. دعم Google Veo 3.1 Fast (أسرع موديل فيديو)
3. نظام اكتشاف تلقائي للموديلات
4. تبديل ذكي بين 5 مزودين
5. تحسين تلقائي للأوصاف
6. نظام Rate Limiting ذكي
7. رسوم بيانية للإحصائيات

🔧 **المزودون المدعومون:**
• Google AI (Gemini 2.5, Imagen 3.0, Veo 3.1)
• OpenAI (GPT-5.0, GPT-5.2, DALL-E 3)
• Stability AI (SD3.5 Large)
• Luma AI (Dream Machine)
• Kling AI (قريباً)

⚡ **النظام الذكي:**
- يرتب الموديلات من الأحدث للأقدم
- يتبدل تلقائياً عند الخطأ (16 محاولة)
- يحسن الأوصاف أوتوماتيكياً
- Rate Limiting لكل مستخدم

💥 **للاستفسارات أو إضافة مميزات:**
👨‍💻 المطور: Ahmed Elsayed
📞 الدعم: @elbashatech
"""
    await update.message.reply_text(about_text, parse_mode='Markdown')

async def limits_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """عرض حدود الاستخدام اليومية"""
    limits_text = """
📊 **حدود الاستخدام اليومية (لكل مستخدم)**

🤖 **الذكاء الاصطناعي:**
💬 المحادثات: 20 رسالة يومياً
🎨 الصور المولدة: 5 صور يومياً
🎬 الفيديوهات: 2 فيديو يومياً

⚡ **نظام Rate Limiting:**
• محادثة: رسالة واحدة كل ثانية (20/دقيقة)
• صور: صورة واحدة كل 2 ثانية (5/دقيقة)
• فيديو: فيديو واحد كل 10 ثواني (2/دقيقة)

📈 **نصائح للاستخدام الأمثل:**
1. استخدم أوصاف واضحة ومفصلة
2. الفيديوهات تستغرق 30 ث - 5 دقائق
3. Veo 3.1 Fast أسرع من Veo العادي
4. استخدم `/myactivity` لتتبع استخدامك

🔄 **التجديد:** تلقائي كل 24 ساعة (توقيت UTC)
"""
    await update.message.reply_text(limits_text, parse_mode='Markdown')

async def handle_broadcast_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """تتبع ردود المستخدمين على الإذاعات"""
    if update.message.reply_to_message and update.message.reply_to_message.text:
        replied_text = update.message.reply_to_message.text
        if "إذاعة من الإدارة:" in replied_text:
            user_id = update.effective_user.id
            user = db.get_user(user_id)
            
            if user:
                db.log_activity(
                    user_id=user_id,
                    action="broadcast_replied",
                    details=f"reply: {update.message.text[:50]}"
                )
                
                admin_message = f"""
🔄 **رد على إذاعة:**
👤 المستخدم: {user['first_name']} (@{user['username'] or 'بدون'})
🆔 المعرف: {user_id}
💬 الرد: {update.message.text[:100]}
"""
                
                for admin_id in ADMIN_IDS:
                    try:
                        await context.bot.send_message(
                            chat_id=admin_id,
                            text=admin_message
                        )
                    except Exception as e:
                        logger.error(f"فشل إرسال إشعار للمشرف {admin_id}: {e}")

def run_bot():
    """تشغيل البوت"""
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    
    if not BOT_TOKEN:
        logger.error("❌ BOT_TOKEN غير معين")
        return
    
    application = Application.builder().token(BOT_TOKEN).build()
    setup_handlers(application)
    
    logger.info(f"🤖 بدأ تشغيل بوت النظام الذكي - الإصدار 5.4")
    logger.info(f"👑 عدد المشرفين: {len(ADMIN_IDS)}")
    logger.info(f"📊 نظام Rate Limiting: مفعل")
    logger.info(f"🎬 Google Veo 3.1: مفعل")
    logger.info(f"🤖 OpenAI GPT-5.0/5.2: مفعل")
    
    db_status = check_database_status()
    logger.info(f"💾 حالة قاعدة البيانات: {db_status}")
    
    users_count = db.get_users_count()
    logger.info(f"👥 عدد المستخدمين المسجلين: {users_count}")
    
    application.run_polling(drop_pending_updates=True)

def main():
    """الدالة الرئيسية"""
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    
    if not BOT_TOKEN:
        logger.error("❌ يرجى تعيين BOT_TOKEN في متغيرات Railway")
        return
    
    logger.info("🚀 بدء تشغيل البوت على Railway (النظام الذكي 5.4)...")
    
    try:
        run_bot()
    except Exception as e:
        logger.error(f"❌ فشل في تشغيل البوت: {e}")
        return

if __name__ == "__main__":
    main()
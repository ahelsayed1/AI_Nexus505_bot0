# database.py - النسخة النهائية مع دعم الإحصائيات المتقدمة
import sqlite3
import logging
from datetime import datetime, timedelta
import os
import json

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_name="bot_database.db"):
        """تهيئة قاعدة البيانات"""
        self.db_name = db_name
        self.init_database()
    
    def get_connection(self):
        """الحصول على اتصال بقاعدة البيانات"""
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """إنشاء الجداول إذا لم تكن موجودة"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # جدول المستخدمين
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    join_date TEXT,
                    message_count INTEGER DEFAULT 0,
                    last_active TEXT,
                    is_admin BOOLEAN DEFAULT 0
                )
                ''')
                
                # جدول الإذاعات
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS broadcasts (
                    broadcast_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    admin_id INTEGER,
                    message_text TEXT,
                    sent_date TEXT,
                    recipients_count INTEGER
                )
                ''')
                
                # جدول سجلات النشاط
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS activity_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    action TEXT,
                    timestamp TEXT,
                    details TEXT
                )
                ''')
                
                # جدول استخدام الذكاء الاصطناعي
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    service_type TEXT,
                    usage_date TEXT,
                    usage_count INTEGER DEFAULT 0,
                    UNIQUE(user_id, service_type, usage_date),
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
                ''')
                
                # جدول محادثات الذكاء الاصطناعي
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_conversations (
                    conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    service_type TEXT,
                    user_message TEXT,
                    ai_response TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
                ''')
                
                # جدول الملفات المولدة (صور وفيديوهات)
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_generated_files (
                    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    file_type TEXT,
                    prompt TEXT,
                    file_url TEXT,
                    thumbnail_url TEXT,
                    created_at TEXT,
                    provider TEXT,
                    model_name TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
                ''')
                
                # جدول إحصائيات المزودين
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS provider_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider_name TEXT,
                    service_type TEXT,
                    request_date TEXT,
                    request_count INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,
                    avg_response_time REAL,
                    UNIQUE(provider_name, service_type, request_date)
                )
                ''')
                
                # إنشاء الفهارس لتحسين الأداء
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_ai_usage_date ON ai_usage(usage_date)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_ai_usage_user ON ai_usage(user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_generated_files_user ON ai_generated_files(user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_user ON ai_conversations(user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_date ON ai_conversations(timestamp)')
                
                conn.commit()
                logger.info("✅ قاعدة البيانات جاهزة مع دعم الإحصائيات المتقدمة")
                
        except Exception as e:
            logger.error(f"❌ خطأ في تهيئة قاعدة البيانات: {e}")
    
    # ==================== دوال المستخدمين ====================
    def add_or_update_user(self, user_id, username, first_name, last_name=None):
        """إضافة أو تحديث مستخدم"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                current_time = datetime.now().isoformat()
                
                cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
                existing_user = cursor.fetchone()
                
                if existing_user:
                    cursor.execute('''
                    UPDATE users 
                    SET username=?, first_name=?, last_name=?, last_active=?
                    WHERE user_id=?
                    ''', (username, first_name, last_name, current_time, user_id))
                    
                    cursor.execute('''
                    UPDATE users 
                    SET message_count = message_count + 1 
                    WHERE user_id = ?
                    ''', (user_id,))
                else:
                    cursor.execute('''
                    INSERT INTO users 
                    (user_id, username, first_name, last_name, join_date, last_active, message_count)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                    ''', (user_id, username, first_name, last_name, current_time, current_time))
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ خطأ في إضافة/تحديث المستخدم: {e}")
            return False
    
    def get_user(self, user_id):
        """الحصول على معلومات مستخدم"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
                user = cursor.fetchone()
                return dict(user) if user else None
        except Exception as e:
            logger.error(f"❌ خطأ في جلب بيانات المستخدم: {e}")
            return None
    
    def get_all_users(self):
        """الحصول على جميع المستخدمين"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users ORDER BY join_date DESC")
                users = cursor.fetchall()
                return [dict(user) for user in users]
        except Exception as e:
            logger.error(f"❌ خطأ في جلب جميع المستخدمين: {e}")
            return []
    
    def get_users_count(self):
        """الحصول على عدد المستخدمين"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM users")
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            logger.error(f"❌ خطأ في جلب عدد المستخدمين: {e}")
            return 0
    
    def get_active_users_count(self, days=7):
        """الحصول على عدد المستخدمين النشطين"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
                cursor.execute('''
                SELECT COUNT(*) FROM users 
                WHERE last_active >= ?
                ''', (cutoff_date,))
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            logger.error(f"❌ خطأ في جلب المستخدمين النشطين: {e}")
            return 0
    
    # ==================== دوال الذكاء الاصطناعي ====================
    
    def save_generated_file(self, user_id, file_type, prompt, file_url, thumbnail_url=None, provider=None, model_name=None):
        """حفظ معلومات الملف المولد مع تفاصيل المزود"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                created_at = datetime.now().isoformat()
                
                cursor.execute('''
                INSERT INTO ai_generated_files 
                (user_id, file_type, prompt, file_url, thumbnail_url, created_at, provider, model_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, file_type, prompt, file_url, thumbnail_url, created_at, provider, model_name))
                
                conn.commit()
                file_id = cursor.lastrowid
                
                # تسجيل النشاط
                self.log_activity(user_id, "generated_file", 
                                f"type={file_type},provider={provider},file_id={file_id}")
                
                return file_id
                
        except Exception as e:
            logger.error(f"❌ خطأ في حفظ الملف المولد: {e}")
            return None
    
    def get_user_generated_files(self, user_id, file_type=None, limit=10):
        """الحصول على الملفات المولدة للمستخدم"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = '''
                SELECT * FROM ai_generated_files 
                WHERE user_id = ?
                '''
                params = [user_id]
                
                if file_type:
                    query += ' AND file_type = ?'
                    params.append(file_type)
                
                query += ' ORDER BY created_at DESC LIMIT ?'
                params.append(limit)
                
                cursor.execute(query, params)
                files = cursor.fetchall()
                return [dict(file) for file in files]
                
        except Exception as e:
            logger.error(f"❌ خطأ في جلب الملفات المولدة: {e}")
            return []
    
    def save_ai_conversation(self, user_id, service_type, user_message, ai_response):
        """حفظ محادثة الذكاء الاصطناعي"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                timestamp = datetime.now().isoformat()
                
                cursor.execute('''
                INSERT INTO ai_conversations 
                (user_id, service_type, user_message, ai_response, timestamp)
                VALUES (?, ?, ?, ?, ?)
                ''', (user_id, service_type, user_message[:500], ai_response[:1000], timestamp))
                
                conn.commit()
                return cursor.lastrowid
                
        except Exception as e:
            logger.error(f"❌ خطأ في حفظ محادثة AI: {e}")
            return None
    
    def get_user_conversations_count_by_day(self, user_id, days=7):
        """الحصول على عدد المحادثات لكل يوم لآخر N أيام"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                result = {}
                for i in range(days-1, -1, -1):
                    day = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                    
                    cursor.execute('''
                    SELECT COUNT(*) FROM ai_conversations 
                    WHERE user_id = ? AND DATE(timestamp) = DATE(?)
                    ''', (user_id, day))
                    
                    count = cursor.fetchone()[0] or 0
                    result[day] = count
                
                return result
                
        except Exception as e:
            logger.error(f"❌ خطأ في جلب عدد المحادثات: {e}")
            return {}
    
    # ==================== دوال الإحصائيات ====================
    
    def get_stats_fixed(self):
        """إحصائيات موثوقة 100%"""
        try:
            stats = {}
            
            # عدد المستخدمين
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM users")
                stats['total_users'] = cursor.fetchone()[0] or 0
            
            # عدد الرسائل
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT SUM(message_count) FROM users")
                total = cursor.fetchone()[0] or 0
                stats['total_messages'] = int(total)
            
            # عدد الإذاعات
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM broadcasts")
                stats['total_broadcasts'] = cursor.fetchone()[0] or 0
            
            # آخر إذاعة
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(broadcast_id) FROM broadcasts")
                stats['last_broadcast_id'] = cursor.fetchone()[0]
            
            # المستخدمين الجدد اليوم
            with self.get_connection() as conn:
                cursor = conn.cursor()
                today = datetime.now().strftime('%Y-%m-%d')
                cursor.execute("SELECT COUNT(*) FROM users WHERE join_date LIKE ?", (f'{today}%',))
                stats['new_users_today'] = cursor.fetchone()[0] or 0
            
            # إجمالي الصور والفيديوهات
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM ai_generated_files WHERE file_type = 'image'")
                stats['total_images'] = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT COUNT(*) FROM ai_generated_files WHERE file_type = 'video'")
                stats['total_videos'] = cursor.fetchone()[0] or 0
            
            # المستخدمين الأكثر نشاطاً
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT first_name, message_count 
                FROM users 
                ORDER BY message_count DESC 
                LIMIT 5
                ''')
                top_users = cursor.fetchall()
                stats['top_users'] = [dict(row) for row in top_users]
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ خطأ في get_stats_fixed: {e}")
            return {
                'total_users': self.get_users_count(),
                'total_messages': 0,
                'total_broadcasts': 0,
                'new_users_today': 0,
                'total_images': 0,
                'total_videos': 0,
                'top_users': []
            }
    
    # ==================== دوال النشاط ====================
    
    def log_activity(self, user_id, action, details=None):
        """تسجيل نشاط المستخدم"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                current_time = datetime.now().isoformat()
                
                cursor.execute('''
                INSERT INTO activity_logs (user_id, action, timestamp, details)
                VALUES (?, ?, ?, ?)
                ''', (user_id, action, current_time, str(details)[:200]))
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ خطأ في تسجيل النشاط: {e}")
            return False
    
    def get_user_activity_summary(self, user_id, days=7):
        """ملخص نشاط المستخدم لآخر N أيام"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
                
                cursor.execute('''
                SELECT action, COUNT(*) as count
                FROM activity_logs
                WHERE user_id = ? AND timestamp >= ?
                GROUP BY action
                ORDER BY count DESC
                ''', (user_id, cutoff_date))
                
                activities = cursor.fetchall()
                return [dict(act) for act in activities]
                
        except Exception as e:
            logger.error(f"❌ خطأ في جلب ملخص النشاط: {e}")
            return []
    
    # ==================== دوال النسخ الاحتياطي ====================
    
    def backup_database(self, backup_name=None):
        """إنشاء نسخة احتياطية من قاعدة البيانات"""
        import shutil
        
        try:
            if backup_name is None:
                backup_name = f"backups/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            
            # إنشاء مجلد backups إذا لم يكن موجوداً
            os.makedirs("backups", exist_ok=True)
            
            shutil.copy2(self.db_name, backup_name)
            logger.info(f"✅ تم إنشاء نسخة احتياطية: {backup_name}")
            
            # تنظيف النسخ القديمة (احتفظ بـ 7 أيام)
            self.cleanup_old_backups()
            
            return backup_name
        except Exception as e:
            logger.error(f"❌ خطأ في إنشاء النسخة الاحتياطية: {e}")
            return None
    
    def cleanup_old_backups(self, days=7):
        """تنظيف النسخ الاحتياطية القديمة"""
        try:
            import os
            from datetime import datetime, timedelta
            
            backup_dir = "backups"
            if not os.path.exists(backup_dir):
                return 0
            
            cutoff = datetime.now() - timedelta(days=days)
            deleted = 0
            
            for file in os.listdir(backup_dir):
                if file.startswith("backup_") and file.endswith(".db"):
                    file_path = os.path.join(backup_dir, file)
                    file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                    
                    if file_time < cutoff:
                        os.remove(file_path)
                        deleted += 1
            
            if deleted > 0:
                logger.info(f"🧹 تم تنظيف {deleted} نسخة احتياطية قديمة")
            
            return deleted
            
        except Exception as e:
            logger.error(f"❌ خطأ في تنظيف النسخ الاحتياطية: {e}")
            return 0

# إنشاء كائن قاعدة بيانات عالمي
db = Database()
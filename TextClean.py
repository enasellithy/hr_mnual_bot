import re
import unicodedata
from typing import Optional

class TextCleaner:
    """طبقة لتنظيف النصوص لدعم لغات متعددة، مع التركيز على العربية والإنجليزية."""
    
    def __init__(self, keep_numbers: bool = False, custom_stopwords: Optional[list] = None):
        """
        تهيئة منظف النصوص.
        
        Args:
            keep_numbers (bool): إذا كان True، يتم الاحتفاظ بالأرقام وإلا يتم إزالتها.
            custom_stopwords (list, optional): قائمة مخصصة بالكلمات غير المفيدة (stop words) للإزالة.
        """
        self.keep_numbers = keep_numbers
        
        # قوائم أولية للكلمات غير المفيدة (يمكن توسيعها وتحسينها)
        self.base_stopwords = {
            'ar': ['في', 'من', 'على', 'إلى', 'أن', 'هذا', 'هذه', 'كان', 'هل', 'أو'],
            'en': ['the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 'and', 'is', 'are']
        }
        
        self.custom_stopwords = custom_stopwords or []
    
    def detect_language(self, text: str) -> str:
        """كشف بسيط للغة بناءً على نطاق الحروف Unicode [citation:4]."""
        arabic_count = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        # يمكن إضافة لغات أخرى هنا
        if arabic_count > len(text) * 0.3:  # إذا كانت نسبة الحروف العربية > 30%
            return 'ar'
        else:
            return 'en'  # الافتراضي، يمكن تحسينه للكشف عن لغات أخرى
    
    def normalize_unicode(self, text: str) -> str:
        """تطبيع حروف Unicode، مثل تحويل الأحرف العربية مع التشكيل إلى أشكالها الأساسية [citation:1]."""
        # تفكيك الحروف ثم إعادة تركيبها بدون علامات التشكيل (للعربية)
        text = unicodedata.normalize('NFKD', text)
        # إزالة علامات التشكيل (Diacritics)
        text = ''.join([char for char in text if not unicodedata.combining(char)])
        return text
    
    def remove_special_chars(self, text: str, lang: str) -> str:
        """إزالة الرموز الخاصة والحفاظ على ما يناسب اللغة [citation:1][citation:10]."""
        if lang == 'ar':
            # الاحتفاظ بالأحرف العربية، الأرقام (إذا مطلوب)، والمسافات
            pattern = r'[^\u0600-\u06FF\s]'
            if self.keep_numbers:
                pattern = r'[^\u0600-\u06FF0-9\s]'
        else:  # الإنجليزية أو اللغات اللاتينية
            # الاحتفاظ بالأحرف اللاتينية، الأرقام (إذا مطلوب)، والمسافات
            pattern = r'[^a-zA-Z\s]'
            if self.keep_numbers:
                pattern = r'[^a-zA-Z0-9\s]'
        
        text = re.sub(pattern, ' ', text)
        # إزالة المسافات الزائدة
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def clean(self, text: str, lang: Optional[str] = None) -> str:
        """
        التنظيف الرئيسي للنص.
        
        Args:
            text (str): النص المدخل.
            lang (str, optional): اللغة. إذا لم تحدد، يتم الكشف تلقائياً.
        
        Returns:
            str: النص النظيف.
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 1. كشف اللغة إذا لم يتم تحديدها
        if not lang:
            lang = self.detect_language(text)
        
        # 2. تحويل إلى أحرف صغيرة (لللغات اللاتينية فقط، حيث أن العربية لا تميز حالة)
        if lang == 'en':
            text = text.lower()
        
        # 3. تطبيع Unicode [citation:1]
        text = self.normalize_unicode(text)
        
        # 4. إزالة الرموز الخاصة
        text = self.remove_special_chars(text, lang)
        
        # 5. تقسيم النص إلى كلمات (Tokenization بسيط)
        words = text.split()
        
        # 6. إزالة الكلمات غير المفيدة (Stop words)
        stopwords_to_remove = self.base_stopwords.get(lang, []) + self.custom_stopwords
        words = [word for word in words if word not in stopwords_to_remove]
        
        # 7. إعادة تجميع الكلمات
        clean_text = ' '.join(words)
        
        return clean_text
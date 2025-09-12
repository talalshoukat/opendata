from typing import Dict
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_RIGHT, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import os

class PDFStyleManager:
    def __init__(self):
        # Get the directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fonts_dir = os.path.join(os.path.dirname(current_dir), 'fonts')
        
        # Register fonts for mixed Arabic-Latin text support
        self._register_fonts(fonts_dir)
        
        # GOSI color scheme
        self.gosi_colors = {
            'primary': '#00004C',  # GOSI Dark Blue (0,0,76)
            'secondary': '#00C100',  # GOSI Green (0,193,0)
            'accent': '#00C100',  # GOSI Green (0,193,0)
            'text': '#212121',  # Dark gray
            'light_gray': '#F5F5F5',  # Light gray background
            'white': '#FFFFFF',  # White
            'table_header': '#00004C',  # Blue headers
            'table_data': '#FFFFFF',  # White data cells
            'table_alt': '#F8F9FA'  # Light gray alternating rows
        }

    def _register_fonts(self, fonts_dir: str):
        """Register Arabic and Latin fonts for mixed text support"""
        try:
            # Register Amiri fonts (supports both Arabic and Latin)
            amiri_regular = os.path.join(fonts_dir, 'Amiri-Regular.ttf')
            amiri_bold = os.path.join(fonts_dir, 'Amiri-Bold.ttf')
            amiri_italic = os.path.join(fonts_dir, 'Amiri-Italic.ttf')
            amiri_bold_italic = os.path.join(fonts_dir, 'Amiri-BoldItalic.ttf')
            
            if os.path.exists(amiri_regular):
                pdfmetrics.registerFont(TTFont('Amiri-Regular', amiri_regular))
            
            if os.path.exists(amiri_bold):
                pdfmetrics.registerFont(TTFont('Amiri-Bold', amiri_bold))
                
            if os.path.exists(amiri_italic):
                pdfmetrics.registerFont(TTFont('Amiri-Italic', amiri_italic))
                
            if os.path.exists(amiri_bold_italic):
                pdfmetrics.registerFont(TTFont('Amiri-BoldItalic', amiri_bold_italic))

        except Exception as e:
            print(f"Warning: Error registering fonts: {e}")

    def convert_digits_to_arabic(self, text: str) -> str:
        """Convert Western digits to Arabic-Indic digits"""
        western = "0123456789"
        arabic_indic = "٠١٢٣٤٥٦٧٨٩"
        trans = str.maketrans(western, arabic_indic)
        return text.translate(trans)

    def reshape_text(self, text: str, language: str = "en") -> str:
        """Reshape text for Arabic if needed"""
        if language == "ar":
            # # Convert digits to Arabic-Indic
            # text = self.convert_digits_to_arabic(text)
            # Reshape Arabic text
            reshaped = arabic_reshaper.reshape(text)
            return get_display(reshaped)
        return text

    def get_font_name(self, language: str = 'en', style: str = 'regular') -> str:
        """Get appropriate font name based on language and style"""
        if language == 'ar':
            # Use Amiri fonts for Arabic text (supports mixed Arabic-Latin)
            if style == 'bold':
                return 'Amiri-Bold' if 'Amiri-Bold' in pdfmetrics.getRegisteredFontNames() else 'Amiri-Regular'
            elif style == 'italic':
                return 'Amiri-Italic' if 'Amiri-Italic' in pdfmetrics.getRegisteredFontNames() else 'Amiri-Regular'
            elif style == 'bold_italic':
                return 'Amiri-BoldItalic' if 'Amiri-BoldItalic' in pdfmetrics.getRegisteredFontNames() else 'Amiri-Regular'
            else:
                return 'Amiri-Regular' if 'Amiri-Regular' in pdfmetrics.getRegisteredFontNames() else 'NotoSansArabic-Regular'
        else:
            # Use Helvetica for Latin-only text
            return 'Helvetica'

    def create_styles(self, language: str = 'en') -> Dict[str, ParagraphStyle]:
        """Create custom paragraph styles with RTL/LTR support and mixed text capability"""
        styles = getSampleStyleSheet()

        # Determine alignment based on language
        if language == 'ar':
            alignment = TA_RIGHT
        else:
            alignment = TA_LEFT

        # Get appropriate font
        font_name = self.get_font_name(language, 'regular')
        bold_font_name = self.get_font_name(language, 'bold')
        italic_font_name = self.get_font_name(language, 'italic')

        custom_styles = {
            'title': ParagraphStyle(
                name='GOSITitle',
                parent=styles['Heading1'],
                fontName=font_name,
                fontSize=24,
                textColor=colors.HexColor(self.gosi_colors['primary']),
                alignment=TA_CENTER,
                spaceAfter=10,
                leading=28
            ),
            'subtitle': ParagraphStyle(
                name='GOSISubtitle',
                parent=styles['Heading2'],
                fontName=font_name,
                fontSize=16,
                textColor=colors.HexColor(self.gosi_colors['secondary']),
                alignment=alignment,
                spaceAfter=12,
                leading=20
            ),
            'body': ParagraphStyle(
                name='GOSIBody',
                parent=styles['Normal'],
                fontName=font_name,
                fontSize=11,
                textColor=colors.HexColor(self.gosi_colors['text']),
                alignment=alignment,
                spaceAfter=6,
                leading=14
            ),
            'body_bold': ParagraphStyle(
                name='GOSIBodyBold',
                parent=styles['Normal'],
                fontName=bold_font_name,
                fontSize=11,
                textColor=colors.HexColor(self.gosi_colors['text']),
                alignment=alignment,
                spaceAfter=6,
                leading=14
            ),
            'body_italic': ParagraphStyle(
                name='GOSIBodyItalic',
                parent=styles['Normal'],
                fontName=italic_font_name,
                fontSize=11,
                textColor=colors.HexColor(self.gosi_colors['text']),
                alignment=alignment,
                spaceAfter=6,
                leading=14
            ),
            'heading1': ParagraphStyle(
                name='GOSIHeading1',
                parent=styles['Heading1'],
                fontName=font_name,
                fontSize=18,
                textColor=colors.HexColor(self.gosi_colors['primary']),
                alignment=alignment,
                spaceAfter=10,
                leading=22
            ),
            'heading2': ParagraphStyle(
                name='GOSIHeading2',
                parent=styles['Heading2'],
                fontName=font_name,
                fontSize=14,
                textColor=colors.HexColor(self.gosi_colors['primary']),
                alignment=alignment,
                spaceAfter=8,
                leading=18
            ),
            'disclaimer': ParagraphStyle(
                name='GOSIDisclaimer',
                parent=styles['Normal'],
                fontName=font_name,
                fontSize=9,
                textColor=colors.HexColor('#666666'),
                alignment=alignment,
                spaceAfter=6,
                leading=12,
                borderWidth=1,
                borderColor=colors.HexColor('#CCCCCC'),
                borderPadding=8,
                backColor=colors.HexColor('#F9F9F9')
            ),
            'header': ParagraphStyle(
                name='GOSIHeader',
                parent=styles['Heading1'],
                fontName=font_name,
                fontSize=18,
                textColor=colors.HexColor(self.gosi_colors['primary']),
                alignment=TA_CENTER,
                spaceAfter=10,
                leading=22
            ),
            'footer': ParagraphStyle(
                name='GOSIFooter',
                parent=styles['Normal'],
                fontName=font_name,
                fontSize=8,
                textColor=colors.HexColor('#666666'),
                alignment=TA_CENTER,
                spaceAfter=0,
                leading=10
            ),
            'table_header': ParagraphStyle(
                name='GOSITableHeader',
                parent=styles['Normal'],
                fontName=bold_font_name,
                fontSize=10,
                textColor=colors.HexColor(self.gosi_colors['white']),
                alignment=TA_CENTER,
                spaceAfter=0,
                leading=12,
                backColor=colors.HexColor(self.gosi_colors['table_header'])
            ),
            'table_data': ParagraphStyle(
                name='GOSITableData',
                parent=styles['Normal'],
                fontName=font_name,
                fontSize=9,
                textColor=colors.HexColor(self.gosi_colors['text']),
                alignment=TA_CENTER,
                spaceAfter=0,
                leading=11
            )
        }

        return custom_styles

    def get_mixed_text_style(self, text: str, base_style: str = 'body') -> str:
        """Determine if text contains Arabic characters and return appropriate style"""
        # Check if text contains Arabic characters
        has_arabic = any('\u0600' <= char <= '\u06FF' for char in text)
        
        if has_arabic:
            return f"{base_style}_arabic" if f"{base_style}_arabic" in self.create_styles('ar') else base_style
        else:
            return base_style
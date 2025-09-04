#!/usr/bin/env python3
"""
Enhanced PDF Report Generator for GOSI Data Analysis
Creates professional PDF reports with GOSI branding, multi-language support, and improved formatting
"""

import os
import io
import base64
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.platypus.frames import Frame
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics import renderPDF
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import logging

# Initialize logger first
logger = logging.getLogger(__name__)

# Try to import Arabic text shaping libraries
try:
    import arabic_reshaper
    from bidi.algorithm import get_display

    ARABIC_SHAPING_AVAILABLE = True
except ImportError:
    ARABIC_SHAPING_AVAILABLE = False
    logger.warning("Arabic text shaping libraries not available. Arabic text may not display properly.")


class EnhancedGOSIReportGenerator:
    """Generate professional PDF reports with enhanced GOSI branding and multi-language support"""

    def __init__(self, logo_path: str = None):
        """
        Initialize the enhanced GOSI report generator

        Args:
            logo_path: Path to the GOSI logo file
        """
        self.logo_path = logo_path or os.path.join(os.path.dirname(__file__), '..', 'logo', 'GOSILogo.png')
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

        # Verify logo exists
        if not os.path.exists(self.logo_path):
            logger.warning(f"GOSI logo not found at {self.logo_path}")
            self.logo_path = None

        # Register Arabic fonts
        self.register_arabic_fonts()

    def register_arabic_fonts(self):
        """Register Arabic fonts for proper Arabic text rendering"""
        try:
            # First, try to register Naskh Arabic font (better for shaped text)
            naskh_font_path = os.path.join(os.path.dirname(__file__), '..', 'fonts', 'NotoNaskhArabic-Regular.ttf')

            if os.path.exists(naskh_font_path):
                try:
                    # Register the Naskh font
                    pdfmetrics.registerFont(TTFont('ArabicFont', naskh_font_path))
                    pdfmetrics.registerFont(TTFont('ArabicFont-Bold', naskh_font_path))
                    pdfmetrics.registerFont(TTFont('ArabicFont-Italic', naskh_font_path))
                    pdfmetrics.registerFont(TTFont('ArabicFont-BoldItalic', naskh_font_path))

                    # Register the font family
                    pdfmetrics.registerFontFamily(
                        'ArabicFont',
                        normal='ArabicFont',
                        bold='ArabicFont-Bold',
                        italic='ArabicFont-Italic',
                        boldItalic='ArabicFont-BoldItalic'
                    )

                    logger.info(f"Successfully registered Naskh Arabic font family: {naskh_font_path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to register Naskh font: {e}")

            # Fallback to Sans Arabic font
            bundled_font_path = os.path.join(os.path.dirname(__file__), '..', 'fonts', 'NotoSansArabic-Regular.ttf')

            if os.path.exists(bundled_font_path):
                try:
                    # Register the regular font
                    pdfmetrics.registerFont(TTFont('ArabicFont', bundled_font_path))

                    # Try to register bold font if available
                    bold_font_path = os.path.join(os.path.dirname(__file__), '..', 'fonts', 'NotoSansArabic-Bold.ttf')
                    if os.path.exists(bold_font_path):
                        pdfmetrics.registerFont(TTFont('ArabicFont-Bold', bold_font_path))
                        logger.info("Registered Arabic Bold font")
                    else:
                        # Fallback to regular font for bold
                        pdfmetrics.registerFont(TTFont('ArabicFont-Bold', bundled_font_path))

                    # Use regular font for italic variants (since we don't have italic Arabic fonts)
                    pdfmetrics.registerFont(TTFont('ArabicFont-Italic', bundled_font_path))
                    pdfmetrics.registerFont(TTFont('ArabicFont-BoldItalic', bundled_font_path))

                    # Register the font family
                    pdfmetrics.registerFontFamily(
                        'ArabicFont',
                        normal='ArabicFont',
                        bold='ArabicFont-Bold',
                        italic='ArabicFont-Italic',
                        boldItalic='ArabicFont-BoldItalic'
                    )

                    logger.info(f"Successfully registered Arabic font family: {bundled_font_path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to register bundled font: {e}")

            # Try to register common Arabic fonts that might be available on the system
            arabic_fonts = [
                # Common Arabic fonts on different systems
                '/System/Library/Fonts/Arial Unicode MS.ttf',  # macOS
                '/System/Library/Fonts/Helvetica.ttc',  # macOS
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',  # Linux
                'C:\\Windows\\Fonts\\arial.ttf',  # Windows
                'C:\\Windows\\Fonts\\calibri.ttf',  # Windows
                'C:\\Windows\\Fonts\\tahoma.ttf',  # Windows (supports Arabic)
                'C:\\Windows\\Fonts\\segoeui.ttf',  # Windows (supports Arabic)
            ]

            for font_path in arabic_fonts:
                if os.path.exists(font_path):
                    try:
                        # Register the font with multiple variants
                        pdfmetrics.registerFont(TTFont('ArabicFont', font_path))
                        pdfmetrics.registerFont(TTFont('ArabicFont-Bold', font_path))
                        pdfmetrics.registerFont(TTFont('ArabicFont-Italic', font_path))
                        pdfmetrics.registerFont(TTFont('ArabicFont-BoldItalic', font_path))

                        # Register the font family
                        pdfmetrics.registerFontFamily(
                            'ArabicFont',
                            normal='ArabicFont',
                            bold='ArabicFont-Bold',
                            italic='ArabicFont-Italic',
                            boldItalic='ArabicFont-BoldItalic'
                        )

                        logger.info(f"Successfully registered Arabic font family: {font_path}")
                        return
                    except Exception as e:
                        logger.warning(f"Failed to register font {font_path}: {e}")
                        continue

            # If no Arabic fonts found, create a fallback solution
            logger.warning("No Arabic fonts found. Using fallback approach for Arabic text.")
            self._create_fallback_arabic_support()

        except Exception as e:
            logger.error(f"Error registering Arabic fonts: {e}")

    def _create_fallback_arabic_support(self):
        """Create a fallback solution for Arabic text when no Arabic fonts are available"""
        try:
            # For now, we'll use a simple approach - convert Arabic to transliterated text
            # This is a temporary solution until we can get proper Arabic fonts
            logger.info("Arabic font fallback: Using transliteration approach")
        except Exception as e:
            logger.error(f"Error creating Arabic fallback: {e}")

    def _shape_arabic_text(self, text: str) -> str:
        """Shape Arabic text for proper rendering with connected characters"""
        if not ARABIC_SHAPING_AVAILABLE:
            return text

        try:
            # Try different approaches for better Arabic text rendering
            # Method 1: Reshape + Bidi (most comprehensive)
            try:
                reshaped_text = arabic_reshaper.reshape(text)
                shaped_text = get_display(reshaped_text)
                logger.info(f"Arabic text shaped (reshape+bidi): '{text}' -> '{shaped_text}'")
                return shaped_text
            except Exception as e1:
                logger.warning(f"Reshape+bidi failed: {e1}")

            # Method 2: Just Bidi (simpler, sometimes works better with ReportLab)
            try:
                bidi_text = get_display(text)
                logger.info(f"Arabic text shaped (bidi only): '{text}' -> '{bidi_text}'")
                return bidi_text
            except Exception as e2:
                logger.warning(f"Bidi only failed: {e2}")

            # Method 3: Just reshaping (fallback)
            try:
                reshaped_only = arabic_reshaper.reshape(text)
                logger.info(f"Arabic text shaped (reshape only): '{text}' -> '{reshaped_only}'")
                return reshaped_only
            except Exception as e3:
                logger.warning(f"Reshape only failed: {e3}")

            # If all methods fail, return original text
            logger.warning("All Arabic text shaping methods failed, returning original text")
            return text

        except Exception as e:
            logger.warning(f"Error in Arabic text shaping: {e}")
            return text

    def _transliterate_arabic(self, text: str) -> str:
        """Simple Arabic to Latin transliteration for fallback"""
        # Basic Arabic to Latin transliteration mapping
        arabic_to_latin = {
            'ا': 'a', 'ب': 'b', 'ت': 't', 'ث': 'th', 'ج': 'j', 'ح': 'h', 'خ': 'kh',
            'د': 'd', 'ذ': 'dh', 'ر': 'r', 'ز': 'z', 'س': 's', 'ش': 'sh', 'ص': 's',
            'ض': 'd', 'ط': 't', 'ظ': 'z', 'ع': 'a', 'غ': 'gh', 'ف': 'f', 'ق': 'q',
            'ك': 'k', 'ل': 'l', 'م': 'm', 'ن': 'n', 'ه': 'h', 'و': 'w', 'ي': 'y',
            'ء': 'a', 'آ': 'aa', 'أ': 'a', 'إ': 'i', 'ؤ': 'w', 'ئ': 'y',
            'ة': 'h', 'ى': 'a', 'لا': 'la'
        }

        result = ""
        for char in text:
            if char in arabic_to_latin:
                result += arabic_to_latin[char]
            else:
                result += char
        return result

    def detect_language(self, text: str) -> str:
        """Detect if text is in Arabic or English"""
        # Simple Arabic detection - look for Arabic characters
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
        if arabic_pattern.search(text):
            return 'ar'
        return 'en'

    def get_localized_text(self, key: str, language: str = 'en') -> str:
        """Get localized text based on language"""
        texts = {
            'en': {
                'title': 'GOSI Data Analysis Report',
                'subtitle': 'General Organization for Social Insurance',
                'report_date': 'Report Date',
                'disclaimer_title': 'IMPORTANT DISCLAIMER',
                'disclaimer_text': """
                <b>Data Analysis Report Disclaimer:</b><br/><br/>

                This report contains data analysis results from the GOSI Open Data platform. 
                The analysis and visualizations are generated based on publicly available data 
                and are intended for informational purposes only.<br/><br/>

                <b>Please note:</b><br/>
                • All data is sourced from GOSI's official open data platform<br/>
                • Analysis results are for informational and research purposes<br/>
                • This report should not be used as the sole basis for business decisions<br/>
                • For official GOSI data and analysis, please refer to authorized GOSI channels<br/><br/>

                The General Organization for Social Insurance (GOSI) is committed to providing 
                transparent and accessible data to support research and analysis.
                """,
                'executive_summary': 'Executive Summary',
                'data_analysis': 'Data Analysis',
                'data_visualization': 'Data Visualization',
                'data_overview': 'Data Overview',
                'total_records': 'Total Records',
                'columns': 'Columns',
                'generated_on': 'Generated on',
                'page': 'Page'
            },
            'ar': {
                'title': 'تقرير تحليل بيانات المؤسسة العامة للتأمينات الاجتماعية',
                'subtitle': 'المؤسسة العامة للتأمينات الاجتماعية',
                'report_date': 'تاريخ التقرير',
                'disclaimer_title': 'إخلاء مسؤولية مهم',
                'disclaimer_text': """
                <b>إخلاء مسؤولية تقرير تحليل البيانات:</b><br/><br/>

                يحتوي هذا التقرير على نتائج تحليل البيانات من منصة البيانات المفتوحة للمؤسسة العامة للتأمينات الاجتماعية. 
                يتم إنتاج التحليل والرسوم البيانية بناءً على البيانات المتاحة للجمهور وتهدف للأغراض الإعلامية فقط.<br/><br/>

                <b>يرجى ملاحظة:</b><br/>
                • جميع البيانات مصدرها منصة البيانات المفتوحة الرسمية للمؤسسة العامة للتأمينات الاجتماعية<br/>
                • نتائج التحليل للأغراض الإعلامية والبحثية<br/>
                • لا ينبغي استخدام هذا التقرير كأساس وحيد لاتخاذ القرارات التجارية<br/>
                • للحصول على البيانات والتحليل الرسمي للمؤسسة العامة للتأمينات الاجتماعية، يرجى الرجوع إلى القنوات المعتمدة<br/><br/>

                المؤسسة العامة للتأمينات الاجتماعية ملتزمة بتوفير بيانات شفافة ومتاحة لدعم البحث والتحليل.
                """,
                'executive_summary': 'الملخص التنفيذي',
                'data_analysis': 'تحليل البيانات',
                'data_visualization': 'تصور البيانات',
                'data_overview': 'نظرة عامة على البيانات',
                'total_records': 'إجمالي السجلات',
                'columns': 'الأعمدة',
                'generated_on': 'تم إنشاؤه في',
                'page': 'صفحة'
            }
        }

        text = texts.get(language, texts['en']).get(key, key)

        # Process Arabic text for proper rendering
        if language == 'ar':
            text = self._process_arabic_text(text, language)

        return text

    def _process_arabic_text(self, text: str, language: str) -> str:
        """Process Arabic text for proper rendering"""
        if language == 'ar':
            if not self._is_arabic_font_available():
                # If Arabic font is not available, use transliteration
                return self._transliterate_arabic(text)
            else:
                # If Arabic font is available, shape the text for proper rendering
                return self._shape_arabic_text(text)
        return text

    def _is_arabic_font_available(self) -> bool:
        """Check if Arabic font is available"""
        try:
            registered_fonts = pdfmetrics.getRegisteredFontNames()
            return 'ArabicFont' in registered_fonts or 'ArabicFont-Bold' in registered_fonts
        except:
            return False

    def create_styles(self, language: str = 'en') -> Dict[str, ParagraphStyle]:
        """Create custom paragraph styles for GOSI theme with language support"""
        styles = getSampleStyleSheet()

        # Determine font based on language
        if language == 'ar':
            # Try to use registered Arabic font, fallback to Helvetica
            try:
                # Check if ArabicFont is registered
                if 'ArabicFont' in pdfmetrics.getRegisteredFontNames():
                    font_name = 'ArabicFont'  # Use the family name, not individual variants
                else:
                    font_name = 'Helvetica'
            except:
                font_name = 'Helvetica'
            alignment = TA_RIGHT
        else:
            font_name = 'Helvetica'
            alignment = TA_LEFT

        # Title style
        title_style = ParagraphStyle(
            'GOSITitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=colors.HexColor(self.gosi_colors['primary']),
            alignment=TA_CENTER,
            spaceAfter=20,
            fontName=font_name  # Use family name, ReportLab will handle bold mapping
        )

        # Subtitle style
        subtitle_style = ParagraphStyle(
            'GOSISubtitle',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor(self.gosi_colors['secondary']),
            alignment=alignment,
            spaceAfter=12,
            fontName=font_name  # Use family name, ReportLab will handle bold mapping
        )

        # Body text style
        body_style = ParagraphStyle(
            'GOSIBody',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor(self.gosi_colors['text']),
            alignment=TA_JUSTIFY if language == 'en' else TA_RIGHT,
            spaceAfter=6,
            fontName=font_name
        )

        # Disclaimer style
        disclaimer_style = ParagraphStyle(
            'GOSIDisclaimer',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#666666'),
            alignment=TA_JUSTIFY if language == 'en' else TA_RIGHT,
            spaceAfter=6,
            fontName=font_name,  # Use base font name, not oblique variant
            borderWidth=1,
            borderColor=colors.HexColor('#CCCCCC'),
            borderPadding=8,
            backColor=colors.HexColor('#F9F9F9')
        )

        # Header style
        header_style = ParagraphStyle(
            'GOSIHeader',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor(self.gosi_colors['primary']),
            alignment=TA_CENTER,
            spaceAfter=10,
            fontName=font_name  # Use family name, ReportLab will handle bold mapping
        )

        return {
            'title': title_style,
            'subtitle': subtitle_style,
            'body': body_style,
            'disclaimer': disclaimer_style,
            'header': header_style
        }

    def create_header_footer(self, canvas, doc, language: str = 'en'):
        """Create header and footer for each page with language support"""
        canvas.saveState()

        # Header with GOSI logo and title
        if self.logo_path and os.path.exists(self.logo_path):
            try:
                # Add logo to header - moved higher up
                logo_width = 1.5 * inch
                logo_height = 0.5 * inch
                canvas.drawImage(
                    self.logo_path,
                    x=50,
                    y=doc.height + doc.topMargin - logo_height + 40,  # Moved up by 30 points
                    width=logo_width,
                    height=logo_height,
                    preserveAspectRatio=True,
                    mask='auto'
                )
            except Exception as e:
                logger.warning(f"Could not add logo to header: {e}")

        # Header title - moved higher up
        # Use appropriate font for language
        if language == 'ar':
            try:
                if 'ArabicFont' in pdfmetrics.getRegisteredFontNames():
                    canvas.setFont('ArabicFont', 14)  # Use family name
                else:
                    canvas.setFont('Helvetica-Bold', 14)
            except:
                canvas.setFont('Helvetica-Bold', 14)
        else:
            canvas.setFont('Helvetica-Bold', 14)

        canvas.setFillColor(colors.HexColor(self.gosi_colors['primary']))
        header_text = self.get_localized_text('title', language)
        canvas.drawRightString(
            doc.width + doc.leftMargin,
            doc.height + doc.topMargin + 10,  # Moved up by 30 points
            header_text
        )

        # Header line - moved higher up
        canvas.setStrokeColor(colors.HexColor(self.gosi_colors['accent']))
        canvas.setLineWidth(2)
        canvas.line(
            doc.leftMargin,
            doc.height + doc.topMargin,  # Moved up by 30 points
            doc.width + doc.leftMargin,
            doc.height + doc.topMargin
        )

        # Footer
        # Use appropriate font for language
        if language == 'ar':
            try:
                if 'ArabicFont' in pdfmetrics.getRegisteredFontNames():
                    canvas.setFont('ArabicFont', 8)  # Use family name
                else:
                    canvas.setFont('Helvetica', 8)
            except:
                canvas.setFont('Helvetica', 8)
        else:
            canvas.setFont('Helvetica', 8)

        canvas.setFillColor(colors.HexColor('#666666'))
        footer_text = f"{self.get_localized_text('generated_on', language)} {datetime.now().strftime('%B %d, %Y at %I:%M %p')} | {self.get_localized_text('page', language)} {doc.page}"
        canvas.drawCentredString(
            doc.width / 2 + doc.leftMargin,
            30,
            footer_text
        )

        # Footer line
        canvas.setStrokeColor(colors.HexColor(self.gosi_colors['accent']))
        canvas.setLineWidth(1)
        canvas.line(
            doc.leftMargin,
            40,
            doc.width + doc.leftMargin,
            40
        )

        canvas.restoreState()

    def convert_plotly_to_image(self, fig, width=800, height=600) -> str:
        """Convert Plotly figure to base64 encoded image"""
        try:
            # Convert plotly figure to image bytes
            img_bytes = fig.to_image(format="png", width=width, height=height)

            # Convert to base64 string
            img_base64 = base64.b64encode(img_bytes).decode()

            # Create temporary file
            temp_path = f"/tmp/plotly_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            with open(temp_path, 'wb') as f:
                f.write(base64.b64decode(img_base64))

            return temp_path
        except Exception as e:
            logger.error(f"Error converting plotly figure to image: {e}")
            return None

    def create_enhanced_data_table(self, df: pd.DataFrame, max_rows: int = 20, language: str = 'en') -> Table:
        """Create a formatted table with GOSI theme - white data cells and blue headers"""
        if df.empty:
            return Paragraph("No data available", self.create_styles(language)['body'])

        # Limit rows for display
        display_df = df.head(max_rows)

        # Prepare table data
        table_data = [list(display_df.columns)]  # Header row

        # Add data rows
        for _, row in display_df.iterrows():
            table_data.append([str(cell) for cell in row.values])

        # Create table
        table = Table(table_data, repeatRows=1)

        # Determine appropriate font for language
        if language == 'ar':
            try:
                if 'ArabicFont' in pdfmetrics.getRegisteredFontNames():
                    header_font = 'ArabicFont'  # Use family name
                    data_font = 'ArabicFont'  # Use family name
                else:
                    header_font = 'Helvetica-Bold'
                    data_font = 'Helvetica'
            except:
                header_font = 'Helvetica-Bold'
                data_font = 'Helvetica'
        else:
            header_font = 'Helvetica-Bold'
            data_font = 'Helvetica'

        # Enhanced GOSI table styling
        table_style = TableStyle([
            # Header styling - Blue background
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.gosi_colors['table_header'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), header_font),
            ('FONTSIZE', (0, 0), (-1, 0), 10),

            # Data styling - White background with alternating light gray
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor(self.gosi_colors['table_data'])),
            ('FONTNAME', (0, 1), (-1, -1), data_font),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor(self.gosi_colors['text'])),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E0E0E0')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),

            # Alternating row colors for better readability
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [
                colors.HexColor(self.gosi_colors['table_data']),
                colors.HexColor(self.gosi_colors['table_alt'])
            ]),
        ])

        table.setStyle(table_style)
        return table

    def generate_enhanced_report(self,
                                 query: str,
                                 data: pd.DataFrame,
                                 description: str,
                                 fig: go.Figure = None,
                                 language: str = 'auto',
                                 output_path: str = None) -> str:
        """
        Generate an enhanced PDF report without technical details

        Args:
            query: Original user query
            data: Query results DataFrame
            description: Natural language description of results
            fig: Plotly figure object
            language: Language for the report ('auto', 'en', 'ar')
            output_path: Output file path (optional)

        Returns:
            Path to generated PDF file
        """
        # Auto-detect language if needed
        if language == 'auto':
            language = self.detect_language(query)

        if output_path is None:
            cwd = os.getcwd()
            report_dir = os.path.join(cwd, "reports")
            os.makedirs(report_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(report_dir, f"gosi_report_{timestamp}.pdf")

        # Create document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=100,
            bottomMargin=72
        )

        # Create styles
        styles = self.create_styles(language)

        # Build story (content)
        story = []

        # Title page
        story.append(Spacer(1, 0.3 * inch))  # Reduced from 0.5*inch
        story.append(Paragraph(self.get_localized_text('title', language), styles['title']))
        story.append(Spacer(1, 0.2 * inch))  # Reduced from 0.3*inch
        story.append(Paragraph(self.get_localized_text('subtitle', language), styles['subtitle']))
        story.append(Spacer(1, 0.1 * inch))  # Reduced from 0.2*inch
        story.append(
            Paragraph(f"{self.get_localized_text('report_date', language)}: {datetime.now().strftime('%B %d, %Y')}",
                      styles['body']))
        story.append(Spacer(1, 0.2 * inch))  # Added spacing instead of PageBreak

        # IMPORTANT DISCLAIMER - FIRST THING AFTER TITLE
        story.append(Paragraph(self.get_localized_text('disclaimer_title', language), styles['header']))
        story.append(Spacer(1, 0.05 * inch))  # Reduced from 0.1*inch

        disclaimer_text = self.get_localized_text('disclaimer_text', language)
        story.append(Paragraph(disclaimer_text, styles['disclaimer']))
        story.append(Spacer(1, 0.2 * inch))  # Reduced from 0.3*inch

        # Executive Summary
        story.append(Paragraph(self.get_localized_text('executive_summary', language), styles['header']))
        story.append(Spacer(1, 0.1 * inch))
        processed_description = self._process_arabic_text(description, language)
        story.append(Paragraph(processed_description, styles['body']))
        story.append(Spacer(1, 0.2 * inch))

        # Data Analysis Section
        story.append(Paragraph(self.get_localized_text('data_analysis', language), styles['subtitle']))
        story.append(Spacer(1, 0.1 * inch))

        # Original Query
        story.append(Paragraph("Original Query:", styles['body']))
        processed_query = self._process_arabic_text(query, language)
        story.append(Paragraph(f'<i>"{processed_query}"</i>', styles['body']))
        story.append(Spacer(1, 0.2 * inch))

        # Data Visualization
        if fig is not None:
            story.append(Paragraph(self.get_localized_text('data_visualization', language), styles['subtitle']))
            story.append(Spacer(1, 0.1 * inch))

            # Convert plotly figure to image
            img_path = self.convert_plotly_to_image(fig)
            if img_path and os.path.exists(img_path):
                try:
                    img = Image(img_path, width=6 * inch, height=4 * inch)
                    story.append(img)
                    story.append(Spacer(1, 0.1 * inch))

                    # Clean up temporary file
                    os.remove(img_path)
                except Exception as e:
                    logger.error(f"Error adding image to PDF: {e}")
                    story.append(Paragraph("Visualization could not be included in PDF", styles['body']))
            else:
                story.append(Paragraph("Visualization could not be generated", styles['body']))

            story.append(Spacer(1, 0.2 * inch))

        # Data Overview
        story.append(Paragraph(self.get_localized_text('data_overview', language), styles['subtitle']))
        story.append(Spacer(1, 0.1 * inch))

        # Data summary
        story.append(Paragraph(f"{self.get_localized_text('total_records', language)}: {len(data)}", styles['body']))
        story.append(Paragraph(f"{self.get_localized_text('columns', language)}: {', '.join(data.columns.tolist())}",
                               styles['body']))
        story.append(Spacer(1, 0.1 * inch))

        # Enhanced data table with GOSI theme
        if not data.empty:
            story.append(self.create_enhanced_data_table(data, language=language))
            story.append(Spacer(1, 0.2 * inch))

        # Build PDF with custom header/footer
        doc.build(story,
                  onFirstPage=lambda canvas, doc: self.create_header_footer(canvas, doc, language),
                  onLaterPages=lambda canvas, doc: self.create_header_footer(canvas, doc, language))

        logger.info(f"Enhanced PDF report generated successfully: {output_path}")
        return output_path


def create_enhanced_gosi_report(query: str,
                                data: pd.DataFrame,
                                description: str,
                                fig: go.Figure = None,
                                language: str = 'auto',
                                output_path: str = None) -> str:
    """
    Convenience function to create an enhanced GOSI-themed PDF report

    Args:
        query: Original user query
        data: Query results DataFrame
        description: Natural language description of results
        fig: Plotly figure object
        language: Language for the report ('auto', 'en', 'ar')
        output_path: Output file path (optional)

    Returns:
        Path to generated PDF file
    """
    generator = EnhancedGOSIReportGenerator()
    return generator.generate_enhanced_report(
        query=query,
        data=data,
        description=description,
        fig=fig,
        language=language,
        output_path=output_path
    )
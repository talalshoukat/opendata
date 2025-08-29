#!/usr/bin/env python3
"""
PDF Report Generator for GOSI Data Analysis
Creates professional PDF reports with GOSI branding and theme
"""

import os
import io
import base64
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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import logging

logger = logging.getLogger(__name__)

class GOSIReportGenerator:
    """Generate professional PDF reports with GOSI branding"""
    
    def __init__(self, logo_path: str = None):
        """
        Initialize the GOSI report generator
        
        Args:
            logo_path: Path to the GOSI logo file
        """
        self.logo_path = logo_path or os.path.join(os.path.dirname(__file__), '..', 'logo', 'GOSILogo.png')
        self.gosi_colors = {
            'primary': '#00004C',      # GOSI Dark Blue (0,0,76)
            'secondary': '#00C100',    # GOSI Green (0,193,0)
            'accent': '#00C100',       # GOSI Green (0,193,0)
            'text': '#212121',         # Dark gray
            'light_gray': '#F5F5F5',   # Light gray background
            'white': '#FFFFFF'         # White
        }
        
        # Verify logo exists
        if not os.path.exists(self.logo_path):
            logger.warning(f"GOSI logo not found at {self.logo_path}")
            self.logo_path = None
    
    def create_styles(self) -> Dict[str, ParagraphStyle]:
        """Create custom paragraph styles for GOSI theme"""
        styles = getSampleStyleSheet()
        
        # Title style
        title_style = ParagraphStyle(
            'GOSITitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=colors.HexColor(self.gosi_colors['primary']),
            alignment=TA_CENTER,
            spaceAfter=20,
            fontName='Helvetica-Bold'
        )
        
        # Subtitle style
        subtitle_style = ParagraphStyle(
            'GOSISubtitle',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor(self.gosi_colors['secondary']),
            alignment=TA_LEFT,
            spaceAfter=12,
            fontName='Helvetica-Bold'
        )
        
        # Body text style
        body_style = ParagraphStyle(
            'GOSIBody',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor(self.gosi_colors['text']),
            alignment=TA_JUSTIFY,
            spaceAfter=6,
            fontName='Helvetica'
        )
        
        # Disclaimer style
        disclaimer_style = ParagraphStyle(
            'GOSIDisclaimer',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#666666'),
            alignment=TA_JUSTIFY,
            spaceAfter=6,
            fontName='Helvetica-Oblique',
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
            fontName='Helvetica-Bold'
        )
        
        return {
            'title': title_style,
            'subtitle': subtitle_style,
            'body': body_style,
            'disclaimer': disclaimer_style,
            'header': header_style
        }
    
    def create_header_footer(self, canvas, doc):
        """Create header and footer for each page"""
        canvas.saveState()
        
        # Header with GOSI logo and title
        if self.logo_path and os.path.exists(self.logo_path):
            try:
                # Add logo to header
                logo_width = 1.5 * inch
                logo_height = 0.5 * inch
                canvas.drawImage(
                    self.logo_path,
                    x=50,
                    y=doc.height + doc.topMargin - logo_height - 10,  # Moved up by 10 points
                    width=logo_width,
                    height=logo_height,
                    preserveAspectRatio=True,
                    mask='auto'  # Enable transparency support
                )
            except Exception as e:
                logger.warning(f"Could not add logo to header: {e}")
        
        # Header title
        canvas.setFont('Helvetica-Bold', 14)
        canvas.setFillColor(colors.HexColor(self.gosi_colors['primary']))
        canvas.drawRightString(
            doc.width + doc.leftMargin,
            doc.height + doc.topMargin - 20,
            "GOSI Data Analysis Report"
        )
        
        # Header line
        canvas.setStrokeColor(colors.HexColor(self.gosi_colors['accent']))
        canvas.setLineWidth(2)
        canvas.line(
            doc.leftMargin,
            doc.height + doc.topMargin - 30,
            doc.width + doc.leftMargin,
            doc.height + doc.topMargin - 30
        )
        
        # Footer
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#666666'))
        canvas.drawCentredString(
            doc.width / 2 + doc.leftMargin,
            30,
            f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')} | Page {doc.page}"
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
    
    def create_data_table(self, df: pd.DataFrame, max_rows: int = 20) -> Table:
        """Create a formatted table from DataFrame"""
        if df.empty:
            return Paragraph("No data available", self.create_styles()['body'])
        
        # Limit rows for display
        display_df = df.head(max_rows)
        
        # Prepare table data
        table_data = [list(display_df.columns)]  # Header row
        
        # Add data rows
        for _, row in display_df.iterrows():
            table_data.append([str(cell) for cell in row.values])
        
        # Create table
        table = Table(table_data, repeatRows=1)
        
        # Style the table
        table_style = TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.gosi_colors['primary'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            
            # Data styling
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ])
        
        table.setStyle(table_style)
        return table
    
    def generate_report(self, 
                       query: str,
                       sql_query: str,
                       data: pd.DataFrame,
                       description: str,
                       visualization_code: str = None,
                       fig: go.Figure = None,
                       output_path: str = None) -> str:
        """
        Generate a complete PDF report
        
        Args:
            query: Original user query
            sql_query: Generated SQL query
            data: Query results DataFrame
            description: Natural language description of results
            visualization_code: Python code for visualization
            fig: Plotly figure object
            output_path: Output file path (optional)
            
        Returns:
            Path to generated PDF file
        """
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
        styles = self.create_styles()
        
        # Build story (content)
        story = []
        
        # Title page
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("GOSI Data Analysis Report", styles['title']))
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("General Organization for Social Insurance", styles['subtitle']))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(f"Report Date: {datetime.now().strftime('%B %d, %Y')}", styles['body']))
        story.append(PageBreak())
        
        # IMPORTANT DISCLAIMER - FIRST THING AFTER TITLE
        story.append(Paragraph("IMPORTANT DISCLAIMER", styles['header']))
        story.append(Spacer(1, 0.1*inch))
        
        disclaimer_text = """
        <b>AI-Generated Report Disclaimer:</b><br/><br/>
        
        This report has been generated using artificial intelligence (OpenAI) technology. 
        While every effort has been made to ensure accuracy, there may be errors or 
        inaccuracies in the data analysis, SQL queries, or visualizations presented.<br/><br/>
        
        <b>Please note:</b><br/>
        • All data analysis and SQL queries are generated by AI and should be verified<br/>
        • Visualizations and interpretations are AI-generated and may contain errors<br/>
        • This report is for informational purposes only and should not be used as the 
        sole basis for business decisions<br/>
        • Users are responsible for validating all results before making any decisions<br/><br/>
        
        For official GOSI data and analysis, please refer to authorized GOSI channels 
        and personnel.
        """
        
        story.append(Paragraph(disclaimer_text, styles['disclaimer']))
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['header']))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(description, styles['body']))
        story.append(Spacer(1, 0.2*inch))
        
        # Query Information
        story.append(Paragraph("Query Information", styles['subtitle']))
        story.append(Spacer(1, 0.1*inch))
        
        # Original Query
        story.append(Paragraph("Original Query:", styles['body']))
        story.append(Paragraph(f'<i>"{query}"</i>', styles['body']))
        story.append(Spacer(1, 0.1*inch))
        
        # SQL Query
        story.append(Paragraph("Generated SQL Query:", styles['body']))
        story.append(Paragraph(f'<font name="Courier">{sql_query}</font>', styles['body']))
        story.append(Spacer(1, 0.2*inch))
        
        # Data Results
        story.append(Paragraph("Data Results", styles['subtitle']))
        story.append(Spacer(1, 0.1*inch))
        
        # Data summary
        story.append(Paragraph(f"Total Records: {len(data)}", styles['body']))
        story.append(Paragraph(f"Columns: {', '.join(data.columns.tolist())}", styles['body']))
        story.append(Spacer(1, 0.1*inch))
        
        # Data table
        if not data.empty:
            story.append(self.create_data_table(data))
            story.append(Spacer(1, 0.2*inch))
        
        # Visualization
        if fig is not None:
            story.append(Paragraph("Data Visualization", styles['subtitle']))
            story.append(Spacer(1, 0.1*inch))
            
            # Convert plotly figure to image
            img_path = self.convert_plotly_to_image(fig)
            if img_path and os.path.exists(img_path):
                try:
                    img = Image(img_path, width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 0.1*inch))
                    
                    # Clean up temporary file
                    os.remove(img_path)
                except Exception as e:
                    logger.error(f"Error adding image to PDF: {e}")
                    story.append(Paragraph("Visualization could not be included in PDF", styles['body']))
            else:
                story.append(Paragraph("Visualization could not be generated", styles['body']))
            
            story.append(Spacer(1, 0.2*inch))
        
        # Technical Details
        if visualization_code:
            story.append(Paragraph("Technical Details", styles['subtitle']))
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph("Visualization Code:", styles['body']))
            story.append(Paragraph(f'<font name="Courier" size="8">{visualization_code}</font>', styles['body']))
            story.append(Spacer(1, 0.2*inch))
        

        
        # Build PDF with custom header/footer
        doc.build(story, onFirstPage=self.create_header_footer, onLaterPages=self.create_header_footer)
        
        logger.info(f"PDF report generated successfully: {output_path}")
        return output_path

def create_gosi_report(query: str,
                      sql_query: str,
                      data: pd.DataFrame,
                      description: str,
                      visualization_code: str = None,
                      fig: go.Figure = None,
                      output_path: str = None) -> str:
    """
    Convenience function to create a GOSI-themed PDF report
    
    Args:
        query: Original user query
        sql_query: Generated SQL query
        data: Query results DataFrame
        description: Natural language description of results
        visualization_code: Python code for visualization
        fig: Plotly figure object
        output_path: Output file path (optional)
        
    Returns:
        Path to generated PDF file
    """
    generator = GOSIReportGenerator()
    return generator.generate_report(
        query=query,
        sql_query=sql_query,
        data=data,
        description=description,
        visualization_code=visualization_code,
        fig=fig,
        output_path=output_path
    )

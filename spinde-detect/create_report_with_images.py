#!/usr/bin/env python3
"""
Enhanced Report Generator with Image Embedding
==============================================

This script creates markdown reports with properly embedded images and generates PDFs
with multiple fallback methods to ensure images are included.
"""

import sys
import argparse
import logging
import shutil
import re
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from project_manager import ProjectManager
from report_generator import ReportGenerator

def create_markdown_with_local_images(md_file: Path, output_file: Path, images_dir: Path):
    """Create a modified markdown file with local image references."""
    
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace absolute paths with local image references
    # Pattern: ![Alt text](/absolute/path/to/image.png)
    def replace_image_path(match):
        alt_text = match.group(1)
        image_path = Path(match.group(2))
        image_name = image_path.name
        return f"![{alt_text}](images/{image_name})"
    
    # Replace both absolute and relative image paths
    content = re.sub(r'!\[(.*?)\]\((.*?\.png)\)', replace_image_path, content)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return output_file

def generate_enhanced_pdf(project_dir: str, subject_id: str):
    """Generate PDF with enhanced image handling."""
    
    logger = logging.getLogger(__name__)
    project_path = Path(project_dir)
    
    # Paths
    subject_report_dir = project_path / 'derivatives' / subject_id / 'spindles' / 'report'
    viz_dir = project_path / 'derivatives' / subject_id / 'spindles' / 'visualizations'
    
    # Find the most recent markdown file
    md_files = list(subject_report_dir.glob('*.md'))
    if not md_files:
        logger.error(f"No markdown files found in {subject_report_dir}")
        return None
    
    latest_md = max(md_files, key=lambda x: x.stat().st_mtime)
    
    # Create images directory and copy files
    images_dir = subject_report_dir / 'images'
    images_dir.mkdir(exist_ok=True)
    
    logger.info("Copying visualization files to report directory...")
    image_files = ['comprehensive_analysis.png', 'temporal_analysis.png', 
                   'frequency_distribution.png', 'summary_statistics.png', 
                   'sensor_layout_2d.png']
    
    for img_file in image_files:
        src = viz_dir / img_file
        dst = images_dir / img_file
        if src.exists():
            shutil.copy2(src, dst)
            logger.info(f"✓ Copied {img_file}")
    
    # Create modified markdown with local image paths
    modified_md = subject_report_dir / f"enhanced_{latest_md.stem}.md"
    create_markdown_with_local_images(latest_md, modified_md, images_dir)
    
    # Generate PDF using the modified markdown
    pdf_file = subject_report_dir / f"enhanced_{latest_md.stem}.pdf"
    
    try:
        import subprocess
        
        # Change to report directory for proper relative paths
        result = subprocess.run([
            'pandoc', str(modified_md.name),
            '-o', str(pdf_file.name),
            '--pdf-engine=pdflatex',
            '--variable', 'geometry:margin=1in',
            '--variable', 'fontsize=11pt',
            '--variable', 'documentclass=article'
        ], cwd=subject_report_dir, capture_output=True, text=True, check=True)
        
        if pdf_file.exists():
            logger.info(f"✓ Enhanced PDF generated: {pdf_file}")
            
            # Clean up temporary markdown
            modified_md.unlink()
            
            # Try to open the PDF
            try:
                subprocess.run(['open', str(pdf_file)], check=False)
            except:
                pass
            
            return pdf_file
        
    except subprocess.CalledProcessError as e:
        logger.error(f"PDF generation failed: {e.stderr}")
        
        # Clean up on failure
        if modified_md.exists():
            modified_md.unlink()
    
    return None

def main():
    """Main function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description="Enhanced report generator with proper image embedding")
    parser.add_argument("--project_dir", required=True, help="Path to project directory")
    parser.add_argument("--subject", required=True, help="Subject ID (e.g., sub-001)")
    
    args = parser.parse_args()
    
    try:
        # Validate inputs
        project_manager = ProjectManager(args.project_dir)
        if not project_manager.validate_subject(args.subject):
            logger.error(f"Subject {args.subject} not found")
            return 1
        
        if not project_manager.subject_results_exist(args.subject):
            logger.error(f"No results found for {args.subject}")
            return 1
        
        # Generate regular report first
        logger.info("Generating regular markdown report...")
        report_generator = ReportGenerator(project_manager)
        success = report_generator.generate_subject_report(args.subject)
        
        if not success:
            logger.error("Failed to generate markdown report")
            return 1
        
        # Generate enhanced PDF
        logger.info("Generating enhanced PDF with embedded images...")
        pdf_file = generate_enhanced_pdf(args.project_dir, args.subject)
        
        if pdf_file:
            logger.info("✅ Success! Enhanced PDF report created with embedded images")
            logger.info(f"📄 PDF location: {pdf_file}")
        else:
            logger.error("❌ Failed to generate enhanced PDF")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
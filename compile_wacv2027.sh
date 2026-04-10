#!/bin/bash
# =============================================================================
# compile_wacv2027.sh — Compile CalibPL WACV 2027 Paper
#
# Usage:
#   ./compile_wacv2027.sh           # compile main paper only
#   ./compile_wacv2027.sh --supp    # compile supplementary too
#   ./compile_wacv2027.sh --all     # compile both + create submission zip
#
# Prerequisites: pdflatex, bibtex (texlive-full recommended)
# =============================================================================

set -e
PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paper/wacv2027"
BUILD_DIR="$PAPER_DIR/build"

mkdir -p "$BUILD_DIR"

compile_main() {
    echo ""
    echo "======================================================="
    echo " Compiling: wacv_paper.tex"
    echo "======================================================="
    cd "$PAPER_DIR"

    # First pass: latex
    pdflatex -interaction=nonstopmode -output-directory="$BUILD_DIR" wacv_paper.tex
    # BibTeX (run from build dir, look in parent for .bib)
    cd "$BUILD_DIR"
    export BIBINPUTS="$PAPER_DIR:$BIBINPUTS"
    bibtex wacv_paper
    cd "$PAPER_DIR"
    # Two more passes for references
    pdflatex -interaction=nonstopmode -output-directory="$BUILD_DIR" wacv_paper.tex
    pdflatex -interaction=nonstopmode -output-directory="$BUILD_DIR" wacv_paper.tex

    echo ""
    echo "✓ Main paper compiled: $BUILD_DIR/wacv_paper.pdf"
    # Count pages
    if command -v pdfinfo &> /dev/null; then
        PAGES=$(pdfinfo "$BUILD_DIR/wacv_paper.pdf" 2>/dev/null | grep Pages | awk '{print $2}')
        echo "  Page count: $PAGES / 8 allowed (main content)"
        if [ "$PAGES" -gt 8 ]; then
            echo "  ⚠ WARNING: Paper exceeds 8 pages! Reduce content before submission."
        else
            echo "  ✓ Within 8-page limit."
        fi
    fi
}

compile_supp() {
    echo ""
    echo "======================================================="
    echo " Compiling: wacv_supplementary.tex"
    echo "======================================================="
    cd "$PAPER_DIR"
    pdflatex -interaction=nonstopmode -output-directory="$BUILD_DIR" wacv_supplementary.tex
    cd "$BUILD_DIR"
    export BIBINPUTS="$PAPER_DIR:$BIBINPUTS"
    bibtex wacv_supplementary || true
    cd "$PAPER_DIR"
    pdflatex -interaction=nonstopmode -output-directory="$BUILD_DIR" wacv_supplementary.tex
    pdflatex -interaction=nonstopmode -output-directory="$BUILD_DIR" wacv_supplementary.tex
    echo "✓ Supplementary compiled: $BUILD_DIR/wacv_supplementary.pdf"
}

create_submission_zip() {
    echo ""
    echo "======================================================="
    echo " Creating WACV 2027 submission ZIP"
    echo "======================================================="
    SUBMISSION_DIR="$PAPER_DIR/wacv2027_submission"
    mkdir -p "$SUBMISSION_DIR"

    # Copy PDFs
    cp "$BUILD_DIR/wacv_paper.pdf" "$SUBMISSION_DIR/paper.pdf"
    [ -f "$BUILD_DIR/wacv_supplementary.pdf" ] && \
        cp "$BUILD_DIR/wacv_supplementary.pdf" "$SUBMISSION_DIR/supplementary.pdf"

    # Copy LaTeX sources (anonymized)
    cp "$PAPER_DIR/wacv_paper.tex" "$SUBMISSION_DIR/"
    cp "$PAPER_DIR/wacv_supplementary.tex" "$SUBMISSION_DIR/"
    cp "$PAPER_DIR/wacv2027_refs.bib" "$SUBMISSION_DIR/"
    cp "$PAPER_DIR/wacv.sty" "$SUBMISSION_DIR/"

    # Copy figures
    mkdir -p "$SUBMISSION_DIR/figures"
    cp -r ../../results/figures/*.png "$SUBMISSION_DIR/figures/" 2>/dev/null || true
    cp -r ../../results/figures/*.pdf "$SUBMISSION_DIR/figures/" 2>/dev/null || true

    # Create the zip
    cd "$PAPER_DIR"
    zip -r "wacv2027_calibpl_submission.zip" "wacv2027_submission/"
    echo "✓ Submission ZIP: $PAPER_DIR/wacv2027_calibpl_submission.zip"

    # Anonymization reminder
    echo ""
    echo "============================================================"
    echo " PRE-SUBMISSION ANONYMIZATION CHECKLIST"
    echo "============================================================"
    echo " Before uploading to OpenReview, verify:"
    echo " [ ] No author names in paper body or LaTeX \author{}"
    echo " [ ] No affiliations in paper"
    echo " [ ] No acknowledgments section"
    echo " [ ] No grant/funding IDs"
    echo " [ ] No personal GitHub/arXiv URLs"
    echo " [ ] Supplementary has same anonymization"
    echo " [ ] Code (if submitted) uses anonymous.4open.science"
    echo " [ ] OpenReview profiles for ALL authors: complete + visible"
    echo " [ ] Paper enrolled >= 1 week before deadline"
    echo " [ ] Paper exactly <= 8 pages (main content, not refs)"
    echo "============================================================"
}

# Parse arguments
COMPILE_SUPP=false
CREATE_ZIP=false

for arg in "$@"; do
    case $arg in
        --supp) COMPILE_SUPP=true ;;
        --all)  COMPILE_SUPP=true; CREATE_ZIP=true ;;
    esac
done

compile_main

if $COMPILE_SUPP; then
    compile_supp
fi

if $CREATE_ZIP; then
    create_submission_zip
fi

echo ""
echo "Done."

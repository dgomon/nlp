python3 -c "
from pypdf import PdfWriter

merger = PdfWriter()
merger.append('../out/maman11.pdf')
merger.append('BPETokenizer.pdf')
merger.write('merged.pdf')
merger.close()
"
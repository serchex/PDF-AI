import pdf

pages = pdf.build_pages("zorro_merged.pdf")

sergio_hits = pdf.find_mentions(pages, "espiricueta")
juan_hits   = pdf.find_mentions(pages, "evelio")

print(sergio_hits)
print(juan_hits)
print(pdf.pdf_text("zorro_mergedp.pdf"))
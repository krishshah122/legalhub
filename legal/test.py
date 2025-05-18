import fitz

doc = fitz.open(r"C:\Users\kriss\Downloads\CompaniesAct2013.pdf")
print(len(doc))  # number of pages
doc.close()

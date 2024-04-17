import csv
import pandas as pd

ppl_no = ['302','303','305','306','307','308','309','310','311','312','313','314','315','345','346','347','348','349','350','352','353','354','355','356','358','359']

# field names
fields = ['Class no.', 'Extraversion', 'Agreeableness', 'Neuroticism', 'Conscientiousness', 'Openness']

df = pd.read_excel(r'C:\Users\DARPAN\Documents\College\6th Semester\BSc Project (DSE6)\Data\list_big5(1).xlsx')

dfi = 0

rows = []
for e in ppl_no:
	while int(df.iloc[dfi,1]) != int(e):
		dfi += 1
	E,A,N,C,O = df.iloc[dfi,8], df.iloc[dfi,9], df.iloc[dfi,10], df.iloc[dfi,11], df.iloc[dfi,12]
	P = e
	cur_row = [P,E,A,N,C,O]
	rows.append(cur_row)


# name of csv file
filename = "y_features2.csv"

# writing to csv file
with open(filename, 'w') as csvfile:
	# creating a csv writer object
	csvwriter = csv.writer(csvfile)

	# writing the fields
	csvwriter.writerow(fields)

	# writing the data rows
	csvwriter.writerows(rows)

csvfile.close()

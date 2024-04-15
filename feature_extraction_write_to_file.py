''' TODO: Y FEATURE VECTOR MODIFY KORE RUN KORTE HOBE '''



import cropping_image
import straight_lines
import find_min_bounding_circles_and_rectangles



ppl_no = ['302','303','305','306','307','308','309','310','311','312','313','314','315','345','346','347','348','349','350','352','353','354','355','356','358','359']
file_no = ['_0006','_0007','_0008','_0009','_0010','_0011','_0012','_0013','_0014','_0015','_0016','_0017','_0018','_0020']

import csv

# field names
fields = ['No. of straight lines', 'Median height', 'Median width', 'Median radius', 'Median rectangularity ratio', 'Median circularity ratio', 'Median rectangular area', 'Median circular area']

# data rows of csv file
# rows = [['Nikhil', 'COE', '2', '9.0'],
# 		['Sanchit', 'COE', '2', '9.1'],
# 		['Aditya', 'IT', '2', '9.3'],
# 		['Sagar', 'SE', '1', '9.5'],
# 		['Prateek', 'MCE', '3', '7.8'],
# 		['Sahil', 'EP', '2', '9.1']]


rows = []
# rows2 = []
for e in ppl_no:
	for f in  file_no:
		my_path = 'C:\\Users\\DARPAN\\Documents\\College\\6th Semester\\BSc Project (DSE6)\\Data\\'
		my_path += e+f+'.png'
		
		print(my_path)
		# cropping_image.get_image(my_path)
		cropping_image.main(my_path) 
		P1 = straight_lines.main()
		P2,P3,P4,P5,P6,P7,P8 = find_min_bounding_circles_and_rectangles.main()
		cur_row = [P1,P2,P3,P4,P5,P6,P7,P8]
		# cur_row2 = [e,f]
		# print('cur_row', cur_row)
		rows.append(cur_row)
		

		



# name of csv file
filename = "x_features2.csv"

# writing to csv file
with open(filename, 'w') as csvfile:
	# creating a csv writer object
	csvwriter = csv.writer(csvfile)

	# writing the fields
	csvwriter.writerow(fields)

	# writing the data rows
	csvwriter.writerows(rows)

csvfile.close()




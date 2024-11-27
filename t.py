import pydicom

ds = pydicom.dcmread('D:/tiya2022/dataset/deidentified_nongated/91/91/IM-0001-0008.dcm')
print (ds[0x0028, 0x0030].value[0])
print (ds[0x0018, 0x9318].value)
print(ds[0x0018, 0x9313].value[0])
print(ds[0x0018, 0x9313].value[1])
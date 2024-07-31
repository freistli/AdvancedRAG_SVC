import os

class Common:
    def __init__(self, docPath):
        self.docPath = docPath
    
    def isFile(self):
        return os.path.isfile(self.docPath)
    
    def isDir(self):
        return os.path.isdir(self.docPath)
    
    def isExcelFile(self):
        isExcel = False
        _, file_extension = os.path.splitext(self.docPath)
        file_extension = file_extension.lower()
        if file_extension == '.xls' or file_extension == '.xlsx':
            isExcel = True
        return isExcel
            
    def isTextFile(self):
        isText = False
        _, file_extension = os.path.splitext(self.docPath)
        file_extension = file_extension.lower()
        if file_extension == '.txt' or file_extension == '.csv':
            isText = True
        return isText
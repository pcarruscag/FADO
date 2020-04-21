#  Copyright 2019-2020, Pedro Gomes.
#
#  This file is part of FADO.
#
#  FADO is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  FADO is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with FADO.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np


# Replace text labels by values
class LabelReplacer:
    def __init__(self,label):
        self._label = label

    def write(self,file,value):
        fid = open(file,"r")
        lines = fid.readlines()
        fid.close()

        if isinstance(value,np.ndarray): value = value[0]

        newlines = []
        for line in lines:
            newlines.append(line.replace(self._label,str(value)))
        #end

        fid = open(file,"w")
        fid.writelines(newlines)
        fid.close()
    #end
#end


# Replace text labels by arrays of values
class ArrayLabelReplacer:
    def __init__(self,label,delim=","):
        self._label = label
        self._delim = delim

    def write(self,file,value):
        fid = open(file,"r")
        lines = fid.readlines()
        fid.close()

        valueStr = ""
        for v in value:
            valueStr += str(v)+self._delim
        valueStr = valueStr.strip(self._delim)

        newlines = []
        for line in lines:
            newlines.append(line.replace(self._label,valueStr))
        #end

        fid = open(file,"w")
        fid.writelines(newlines)
        fid.close()
    #end
#end


# Read or write "delim"-separated values in front of a label (pre-string)
class PreStringHandler:
    def __init__(self,label,delim=","):
        self._label = label
        self._delim = delim

    def read(self,file):
        fid = open(file,"r")
        lines = fid.readlines()
        fid.close()

        for line in lines:
            if line.startswith(self._label):
                data = line.lstrip(self._label).strip().split(self._delim)
            #end
        #end

        size = len(data)
        if size==1: return float(data[0])

        value = np.ndarray((size,))
        for i in range(size):
            value[i] = float(data[i])
        
        return value
    #end

    def write(self,file,value):
        fid = open(file,"r")
        lines = fid.readlines()
        fid.close()

        # make scalars iterable
        if isinstance(value,float) or isinstance(value,int):
            value = [value]

        newLine = ""
        for i in range(len(lines)):
            if lines[i].startswith(self._label):
                if not newLine:
                    newLine += self._label
                    for val in value:
                        newLine += str(val)+self._delim
                    newLine = newLine[0:-len(self._delim)]+"\n"
                #end
                lines[i] = newLine
            #end
        #end

        fid = open(file,"w")
        fid.writelines(lines)
        fid.close()
    #end
#end


# Read from a table-like file
class TableReader:
    # use row/col=None to return entire columns/rows, and -1 for "last"
    def __init__(self,row=0,col=0,start=(0,0),end=(None,None),delim=""):
        self._row = row
        self._col = col
        self._end = end
        self._start = start
        self._delim = delim

    def read(self,file):
        fid = open(file,"r")
        lines = fid.readlines()
        fid.close()

        # skip header and footer rows
        lines = lines[self._start[0]:self._end[0]]
        numRow = len(lines)

        # process lines
        data = None
        numCol = 0
        for (line,row) in zip(lines,range(numRow)):
            for char in self._delim:
                line = line.replace(char," ")

            tmp = line.strip().split()[self._start[1]:self._end[1]]
            
            if numCol == 0:
                numCol = len(tmp)
                data = np.ndarray((numRow,numCol))
            elif numCol != len(tmp):
                raise RuntimeError("Data is not in table format.")
            #end

            for col in range(numCol):
                data[row,col] = float(tmp[col])
        #end

        if self._row == None:
            if self._col == None:
                return data
            else:
                return data[:,self._col]
            #end
        else:
            if self._col == None:
                return data[self._row,:]
            else:
                return data[self._row,self._col]
            #end
        #end
    #end
#end


# Write to a table-like file
class TableWriter:
    def __init__(self,delim="  ",start=(0,0),end=(None,None),delimChars=""):
        self._end = end
        self._start = start
        self._delim = delim
        self._delimChars = delimChars

    def write(self,file,values):
        # load file
        fid = open(file,"r")
        lines = fid.readlines()
        fid.close()

        # check if the values are remotely compatible with the file
        if len(lines) < values.shape[0]: return # "soft fail"

        # keep top, bottom, left, and right the same
        newLines = lines[0:self._start[0]]
        footerLines = []
        if self._end[0] is not None: footerLines = lines[self._end[0]:]

        # skip header and footer rows
        lines = lines[self._start[0]:self._end[0]]
        if lines[-1].strip() is "": lines = lines[0:-1]

        if len(lines) != values.shape[0]:
            raise RuntimeError("Data and file have different number of rows.")
        numCol = values.size/values.shape[0]

        # process lines
        for (line,row) in zip(lines,values):
            for char in self._delimChars:
                line = line.replace(char," ")

            tmp = line.strip().split()

            if numCol != len(tmp[self._start[1]:self._end[1]]):
                raise RuntimeError("Data and file have different number of columns.")
            #end

            # reconstruct left and right parts
            newLine = ""
            for string in tmp[0:self._start[1]]:
                newLine += string+self._delim

            # handle case where row is not iterable
            if values.ndim==1: row=[row]

            for val in row:
                newLine += str(val)+self._delim

            if self._end[1] is not None:
                for string in tmp[self._end[1]:]:
                    newLine += string+self._delim

            newLines.append(newLine.strip()+"\n")
        #end

        # write file
        fid = open(file,"w")
        fid.writelines(newLines+footerLines)
        fid.close()
    #end
#end

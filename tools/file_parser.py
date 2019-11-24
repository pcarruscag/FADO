#  Copyright 2019, Pedro Gomes.
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


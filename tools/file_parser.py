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


# Replace text labels by values
class LabelReplacer:
    def __init__(self,label):
        self._label = label

    def write(self,file,value):
        fid = open(file,"r")
        lines = fid.readlines()
        fid.close()

        newlines = []
        for line in lines:
            newlines.append(line.replace(self._label,str(value)))
        #end

        fid = open(file,"w")
        fid.writelines(newlines)
        fid.close()
    #end
#end

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

import os
import shutil
import subprocess as sp


# Class to define an execution of an external code
class ExternalRun:
    def __init__(self,dir="",command=""):
        self._dataFiles = []
        self._confFiles = []
        self._workDir = dir
        self._command = command
        self._process = None
        self._parameters = []

    def addData(self,file):
        self._dataFiles.append(file)

    def addConfig(self,file):
        self._confFiles.append(file)

    def addParameter(self,param):
        self._parameters.append(param)

    def setWorkDir(self,dir):
        self._workDir = dir

    def setCommand(self,command):
        self._command = command

    def initialize(self):
        os.mkdir(self._workDir)
        for file in self._dataFiles:
            shutil.copy(file,self._workDir)

        for file in self._confFiles:
            shutil.copy(file,self._workDir)
            for param in self._parameters:
                param.write(os.path.join(self._workDir,file))

        self._process = sp.Popen(self._command,cwd=self._workDir,
                        shell=True,stdout=sp.PIPE,stderr=sp.PIPE)
    #end

    def run(self,timeout=None):
        return self._process.wait(timeout)

    def poll(self):
        return self._process.poll()

    def finalize(self):
        pass

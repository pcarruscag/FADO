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

import os
import shutil
import subprocess as sp


# Class to define an execution of an external code
class ExternalRun:
    def __init__(self,dir,command,useSymLinks=False):
        self._dataFiles = []
        self._confFiles = []
        self._expectedFiles = []
        self._workDir = dir
        self._command = command
        self._symLinks = useSymLinks
        self._maxTries = 1
        self._numTries = 0
        self._process = None
        self._variables = set()
        self._parameters = []
        self._stdout = None
        self._stderr = None
        self.finalize()

    def _addAbsoluteFile(self,file,flist):
        file = os.path.abspath(file)
        if not os.path.isfile(file):
            raise ValueError("File '"+file+"' not found.")
        flist.append(file)

    def addData(self,file,location="auto"):
        if location is "relative":
            self._dataFiles.append(file)
        else:
            try:
                self._addAbsoluteFile(file,self._dataFiles)
            except:
                if location is "absolute": raise
                # in "auto" mode, if absolute fails consider relative
                else: self._dataFiles.append(file)
            #end
        #end
    #end

    def addConfig(self,file):
        # config files are always assumed to be absolute
        self._addAbsoluteFile(file,self._confFiles)

    def addParameter(self,param):
        self._parameters.append(param)

    def addExpected(self,file):
        # consider always relative
        self._expectedFiles.append(os.path.join(self._workDir,file))

    def setMaxTries(self,num):
        self._maxTries = num

    def getParameters(self):
        return self._parameters

    def updateVariables(self,variables):
        self._variables.update(variables)

    def initialize(self):
        if self._isIni: return

        os.mkdir(self._workDir)
        for file in self._dataFiles:
            target = os.path.join(self._workDir,os.path.basename(file))
            (shutil.copy,os.symlink)[self._symLinks](os.path.abspath(file),target)

        for file in self._confFiles:
            target = os.path.join(self._workDir,os.path.basename(file))
            shutil.copy(file,target)
            for par in self._parameters:
                par.writeToFile(target)
            for var in self._variables:
                var.writeToFile(target)

        self._createProcess()
        self._isIni = True
        self._isRun = False
        self._numTries = 0
    #end

    def _createProcess(self):
        self._stdout = open(os.path.join(self._workDir,"stdout.txt"),"w")
        self._stderr = open(os.path.join(self._workDir,"stderr.txt"),"w")

        self._process = sp.Popen(self._command,cwd=self._workDir,
                        shell=True,stdout=self._stdout,stderr=self._stderr)
    #end

    # on failure this function recurses up to "maxTries" times
    def run(self,timeout=None):
        if not self._isIni:
            raise RuntimeError("Run was not initialized.")
        if self._numTries == self._maxTries:
            raise RuntimeError("Run failed.")
        if self._isRun:
            return self._retcode

        self._retcode = self._process.wait(timeout)
        self._numTries += 1

        if not self._success():
            self.finalize()
            self._createProcess()
            self._isIni = True
            return self.run(timeout)
        #end

        self._numTries = 0
        self._isRun = True
        return self._retcode
    #end

    # again we recurse up to "maxTries" times
    def poll(self):
        if not self._isIni:
            raise RuntimeError("Run was not initialized.")
        if self._numTries == self._maxTries:
            raise RuntimeError("Run failed.")
        if self._isRun:
            return self._retcode

        if self._process.poll() is not None:
            self._numTries += 1

            if not self._success():
                self.finalize()
                self._createProcess()
                self._isIni = True
                return self.poll()
            #end

            self._numTries = 0
            self._retcode = self._process.returncode
            self._isRun = True
        #end

        return self._retcode
    #end

    def isIni(self):
        return self._isIni

    def isRun(self):
        return self._isRun

    # reset "lazy" flags
    def finalize(self):
        try:
            self._stdout.close()
            self._stderr.close()
        except:
            pass
        self._isIni = False
        self._isRun = False
        self._retcode = -100
    #end

    # check whether expected files were created
    def _success(self):
        for file in self._expectedFiles:
            if not os.path.isfile(file): return False
        return True
    #end
#end

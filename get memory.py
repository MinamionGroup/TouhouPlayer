def getRunningProcessExePathByName_win32(name):
  from ctypes import windll, POINTER, pointer, Structure, sizeof
  from ctypes import c_long , c_int , c_uint , c_char , c_ubyte , c_char_p , c_void_p

  class PROCESSENTRY32(Structure):
      _fields_ = [ ( 'dwSize' , c_uint ) ,
                  ( 'cntUsage' , c_uint) ,
                  ( 'th32ProcessID' , c_uint) ,
                  ( 'th32DefaultHeapID' , c_uint) ,
                  ( 'th32ModuleID' , c_uint) ,
                  ( 'cntThreads' , c_uint) ,
                  ( 'th32ParentProcessID' , c_uint) ,
                  ( 'pcPriClassBase' , c_long) ,
                  ( 'dwFlags' , c_uint) ,
                  ( 'szExeFile' , c_char * 260 ) ,
                  ( 'th32MemoryBase' , c_long) ,
                  ( 'th32AccessKey' , c_long ) ]

  class MODULEENTRY32(Structure):
      _fields_ = [ ( 'dwSize' , c_long ) ,
                  ( 'th32ModuleID' , c_long ),
                  ( 'th32ProcessID' , c_long ),
                  ( 'GlblcntUsage' , c_long ),
                  ( 'ProccntUsage' , c_long ) ,
                  ( 'modBaseAddr' , c_long ) ,
                  ( 'modBaseSize' , c_long ) ,
                  ( 'hModule' , c_void_p ) ,
                  ( 'szModule' , c_char * 256 ),
                  ( 'szExePath' , c_char * 260 ) ]

  TH32CS_SNAPPROCESS = 2
  TH32CS_SNAPMODULE = 0x00000008

  ## CreateToolhelp32Snapshot
  CreateToolhelp32Snapshot= windll.kernel32.CreateToolhelp32Snapshot
  CreateToolhelp32Snapshot.reltype = c_long
  CreateToolhelp32Snapshot.argtypes = [ c_int , c_int ]
  ## Process32First
  Process32First = windll.kernel32.Process32First
  Process32First.argtypes = [ c_void_p , POINTER( PROCESSENTRY32 ) ]
  Process32First.rettype = c_int
  ## Process32Next
  Process32Next = windll.kernel32.Process32Next
  Process32Next.argtypes = [ c_void_p , POINTER(PROCESSENTRY32) ]
  Process32Next.rettype = c_int
  ## CloseHandle
  CloseHandle = windll.kernel32.CloseHandle
  CloseHandle.argtypes = [ c_void_p ]
  CloseHandle.rettype = c_int
  ## Module32First
  Module32First = windll.kernel32.Module32First
  Module32First.argtypes = [ c_void_p , POINTER(MODULEENTRY32) ]
  Module32First.rettype = c_int

  hProcessSnap = c_void_p(0)
  hProcessSnap = CreateToolhelp32Snapshot( TH32CS_SNAPPROCESS , 0 )

  pe32 = PROCESSENTRY32()
  pe32.dwSize = sizeof( PROCESSENTRY32 )
  ret = Process32First( hProcessSnap , pointer( pe32 ) )
  path = None

  while ret :
      if name + ".exe" == pe32.szExeFile:
          hModuleSnap = c_void_p(0)
          me32 = MODULEENTRY32()
          me32.dwSize = sizeof( MODULEENTRY32 )
          hModuleSnap = CreateToolhelp32Snapshot( TH32CS_SNAPMODULE, pe32.th32ProcessID )

          ret = Module32First( hModuleSnap, pointer(me32) )
          path = me32.szExePath
          CloseHandle( hModuleSnap )
          if path:
              break
      ret = Process32Next( hProcessSnap, pointer(pe32) )
  CloseHandle( hProcessSnap )
  return path

print(getRunningProcessExePathByName_win32('ShareX'))
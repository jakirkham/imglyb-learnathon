import os
import pathlib
import subprocess
import tempfile
import time
import threading

tmp_dir = tempfile.mkdtemp()

import jnius_config
jnius_config.add_classpath( tmp_dir )


import dask.array as da
import imglyb
import imglyb.types
import imglyb.util
from jnius import PythonJavaClass, java_method, autoclass, JavaException, cast

chunk_list_tmp = []

IntUnsafe = autoclass('net.imglib2.img.basictypelongaccess.unsafe.IntUnsafe')
Cell      = autoclass('net.imglib2.img.cell.Cell')

class LazyCellImgGetFromDaskArray(PythonJavaClass):
    __javainterfaces__ = ['net/imglib2/img/cell/LazyCellImg$Get']

    def __init__(self, dask_array):
        super(LazyCellImgGetFromDaskArray, self).__init__()
        self.dask_array = dask_array
        self.slices = da.core.slices_from_chunks(self.dask_array.chunks)
        

    @java_method('(J)Ljava/lang/Object;', name='get')
    def get(self, index):
        chunk = self.dask_array[self.slices[index]].compute()
        # TODO: instead of adding chunk to global list, pass reference to wrapper around IntUnsafe (or any other unsafe access)
        chunk_list_tmp.append(chunk)
        address = chunk.ctypes.data
        # TODO: replace 2nd chunk.shape by actual min of chunk
        # TODO: do we need to use chunk.shape[::-1] instead?
        return Cell(chunk.shape, chunk.shape, IntUnsafe(address))

make_cell_grid_code = """
import net.imglib2.img.cell.Cell;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.img.cell.LazyCellImg;
import net.imglib2.img.cell.LazyCellImg.Get;
import net.imglib2.type.NativeType;

public class Helpers {

    public static CellGrid makeGrid( long[] dims, int[] cellSize ) {
        return new CellGrid( dims, cellSize );
    }

    public static < T extends NativeType< T >, A > LazyCellImg< T, A > lazyCellImg( final CellGrid grid, final T type, final Get< Cell< A > > get )
    {
        return new LazyCellImg<>( grid, type, get );
    }
    
}
"""

fp = pathlib.Path( tmp_dir ) / 'Helpers.java'
print( tmp_dir )
with open( fp, 'w' ) as f:
    f.write( make_cell_grid_code )

javac = pathlib.Path( os.environ[ 'JAVA_HOME' ] ) / 'bin' / 'javac'
proc = subprocess.run( 
    [ javac, '-cp', jnius_config.split_char.join( jnius_config.get_classpath() ), fp ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE)
if proc.returncode != 0:
    print("Failed!")
    print ( proc.stderr )
    

LazyCellImg = autoclass('net.imglib2.img.cell.LazyCellImg')
CellGrid    = autoclass('net.imglib2.img.cell.CellGrid')
Helpers     = autoclass('Helpers')
ARGBType    = autoclass('net.imglib2.type.numeric.ARGBType')

chunks = [10, 20, 30]
shape  = [100, 200, 290]
array  = da.random.randint(low=0, high=2**31, size=shape, chunks=chunks)
slices = da.core.slices_from_chunks(array.chunks)

get = LazyCellImgGetFromDaskArray(array)
get.get(0)

grid = Helpers.makeGrid(shape[::-1], chunks[::-1])
img  = Helpers.lazyCellImg(grid, ARGBType(), get)

print("Going to open bdv window")
bdv = imglyb.util.BdvFunctions.show(img, 'random')
print("Showing bdv")
vp = bdv.getBdvHandle().getViewerPanel()

# Keep Python running until user closes Bdv window
check = autoclass( 'net.imglib2.python.BdvWindowClosedCheck' )()
frame = cast( 'javax.swing.JFrame', autoclass( 'javax.swing.SwingUtilities' ).getWindowAncestor( vp ) )
frame.addWindowListener( check )

def sleeper():
	while check.isOpen():
		time.sleep( 0.1 )
                

t = threading.Thread( target=sleeper )
t.start()
t.join()
print(len(chunk_list_tmp))

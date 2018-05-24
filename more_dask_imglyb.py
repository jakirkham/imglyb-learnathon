import numpy as np
import os
import pathlib
import subprocess
import tempfile
import time
import threading

tmp_dir = tempfile.mkdtemp()

import jnius_config
jnius_config.add_classpath( tmp_dir )


import dask.array
import dask.array.image
import dask.array as da
import imglyb
import imglyb.types
import imglyb.util
from jnius import PythonJavaClass, java_method, autoclass, JavaException, cast

chunks_dict = {}

IntUnsafe = autoclass('net.imglib2.img.basictypelongaccess.unsafe.IntUnsafe')
Cell      = autoclass('net.imglib2.img.cell.Cell')

make_cell_grid_code = """
import java.util.Arrays;

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

    public static long[] cellMin( final CellGrid grid, long index )
    {
        long[] min = new long[ grid.numDimensions() ];
        grid.getCellGridPositionFlat( index, min );
        Arrays.setAll( min, d -> min[ d ] * grid.cellDimension( d ) );
//        for ( int d = 0; d < min.length; ++d )
//            min[ d ] *= grid.cellDimension( d );
        return min;
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

Helpers = autoclass('Helpers')

class LazyCellImgGetFromDaskArray(PythonJavaClass):
    __javainterfaces__ = ['net/imglib2/img/cell/LazyCellImg$Get']

    def __init__(self, dask_array, cell_grid):
        super(LazyCellImgGetFromDaskArray, self).__init__()
        self.dask_array = dask_array
        self.slices     = da.core.slices_from_chunks(self.dask_array.chunks)
        self.cell_grid  = cell_grid


    @java_method('(J)Ljava/lang/Object;', name='get')
    def get(self, index):
        if not index in chunks_dict:
            chunk = np.ascontiguousarray(self.dask_array[self.slices[index]].compute())
            chunks_dict[index] = chunk
        else:
            chunk = chunks_dict[index]
        # TODO: instead of adding chunk to global list, pass reference to wrapper around IntUnsafe (or any other unsafe access)
        address = chunk.ctypes.data
        # TODO: replace 2nd chunk.shape by actual min of chunk
        # TODO: do we need to use chunk.shape[::-1] instead?
        #print("Getting chunk", chunk.shape, index, address)
        return Cell(chunk.shape[::-1], Helpers.cellMin(self.cell_grid, index), IntUnsafe(address))


LazyCellImg = autoclass('net.imglib2.img.cell.LazyCellImg')
CellGrid    = autoclass('net.imglib2.img.cell.CellGrid')
ARGBType    = autoclass('net.imglib2.type.numeric.ARGBType')
UnsignedIntType    = autoclass('net.imglib2.type.numeric.integer.UnsignedIntType')

# chunks = [2, 3, 4]
# shape  = [4, 6, 8]


chunks = (20, 30)


array = da.image.imread("/Users/kirkhamj/Developer/Jupyter/imglyb-learnathon/notebooks/basics/data/emdata.jpg")[0]
array = array.astype(np.uint32).rechunk(chunks)

shape = array.shape

print(shape)
print(chunks)

# array = da.arange(shape[0], chunks=chunks[0], dtype=np.uint32)
# array = array[:, None].repeat(shape[1], axis=1)
# array = array[:, None].repeat(shape[2], axis=2)
# array = array.rechunk(chunks)
#array  = da.random.randint(low=0, high=2**31, size=shape, chunks=chunks).astype(np.uint32)
slices = da.core.slices_from_chunks(array.chunks)

grid = Helpers.makeGrid(shape[::-1], chunks[::-1])
print("slices")
get  = LazyCellImgGetFromDaskArray(array, grid)
img  = Helpers.lazyCellImg(grid, UnsignedIntType(), get)

BdvOptions = autoclass('bdv.util.BdvOptions')

print("Going to open bdv window")
bdv = imglyb.util.BdvFunctions.show(img, 'random', BdvOptions.options().is2D())
bdv.getBdvHandle().getSetupAssignments().getMinMaxGroups().get(0).setRange(0, 255)
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
#print(len(chunk_list_tmp))

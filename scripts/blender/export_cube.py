from figo_common import module, environ
import figo_common
import random
import os
import numpy as np


# Run In Blender
if module.module_exists("bpy"):
    from blender_figo import objects as bobjects
    from blender_figo import file as bfile
    from blender_figo import batch_process
    from figo_common.math import transform_np
    import bpy

    batch_process.startSelfSupervising()


    def process(filename=None):
        if filename:
            size = np.loadtxt(filename)
            basename = os.path.basename(filename)
        else:
            size = np.random.rand(1) * 10
            basename = 'random'
        sphere = bobjects.create_sphere(radius=1)
        sphere.scale = (size, size, size)
        bpy.context.view_layer.update()
        matrix = np.array(sphere.matrix_world)


        output_path = batch_process.acquireArg("output_path", './temp_process')
        output_path = os.path.join(output_path, 'output')

        bfile.exportFile(os.path.join(output_path, basename + '.obj'))
        np.savetxt(os.path.join(output_path, basename), matrix)

    files = batch_process.acquireFileList()
    if len(files) == 0:
        process()
    else:
        for f in files:
            bfile.newFile(process, f)




# Run In Terminal to Launch Multi-process Batchmode
else:
    import argparse
    from blender_figo import batch_process

    parser = argparse.ArgumentParser(description='Export Random size Cubes.')
    parser.add_argument('--output_path', '-o', type=str, default='./process', help='Output path')
    parser.add_argument('--count', '-c', type=int, default=100, help='Cube count')
    parser.add_argument('--process_count', '-p', type=int, default=4, help='Process count')
    parser.add_argument('--batch_size', '-b', type=int, default=5, help='Batch size')

    args = parser.parse_args()
    output_path = args.output_path
    count = args.count
    process_count = args.process_count
    batch_size = args.batch_size

    filelist = []
    for i in range(count):
        path = os.path.join(output_path, 'input', f'{i}.txt')
        os.makedirs(os.path.join(output_path, 'input'), exist_ok=True)
        np.savetxt(os.path.join(path), np.random.rand(1) * 10)
        filelist.append(path)
    
    batch_process.batchProcess(os.path.abspath('./'), __file__, filelist, batch_size=batch_size, process_count=process_count, args={'output_path': output_path})

import bpy

def load_frame(frame):
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()

def view_update():
    bpy.context.view_layer.update()

def start_frame(frame=None):
    if frame is not None:
        bpy.context.scene.frame_start = frame
    start_frame = bpy.context.scene.frame_start
    return start_frame

def end_frame(frame=None):
    if frame is not None:
        bpy.context.scene.frame_end = frame
    end_frame = bpy.context.scene.frame_end
    return end_frame
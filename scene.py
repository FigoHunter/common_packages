import bpy

def load_frame(frame):
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()

def start_frame():
    start_frame = bpy.context.scene.frame_start
    return start_frame

def end_frame():
    end_frame = bpy.context.scene.frame_end
    return end_frame
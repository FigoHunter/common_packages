import os
from typing import Literal
import pymel.core as pm


def save_scene(path):
    import pymel.core as pm
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pm.saveAs(os.path.join(os.path.abspath('.'),path), force=True)


def new_scene(*, force=True, unit:Literal['cm', 'm', 'mm', 'km', 'in', 'ft', 'yd'] = 'm'):
    """
    Create a new scene in Maya with the specified unit.
    """
    if pm.sceneName() and not force:
        return
    pm.newFile(force=force)
    pm.currentUnit(linear=unit)

def import_fbx(path, namespace=None, fill_timeline=True, merge_namespaces=False):
    """
    Import an FBX file into the current Maya scene.
    
    Args:
        path: Path to the FBX file
        namespace: Namespace for imported objects
        fill_timeline: If True, set timeline range from FBX. Default True.
        merge_namespaces: If True, merge with existing objects in namespace. Default False.
    """
    import pymel.core as pm
    
    # 确保 FBX 插件已加载
    if not pm.pluginInfo('fbxmaya', query=True, loaded=True):
        pm.loadPlugin('fbxmaya')
    
    # 设置 FBX 导入选项
    pm.mel.eval(f'FBXImportFillTimeline -v {"true" if fill_timeline else "false"};')
    
    # 转义路径中的反斜杠
    path = path.replace('\\', '/')
    
    ns_name = namespace.lstrip(':') if namespace else None
    
    if namespace and not merge_namespaces:
        # 普通导入：用 pm.namespace(set=) + FBXImport
        # 创建命名空间（如果不存在）
        if not pm.namespace(exists=ns_name):
            pm.namespace(add=ns_name)
        
        # 设置当前命名空间
        pm.namespace(set=ns_name)
        
        # 用 FBXImport 导入
        cmd = f'FBXImport -file "{path}";'
        print(f"[import_fbx] Namespace set to '{ns_name}', executing: {cmd}")
        pm.mel.eval(cmd)
        
        # 恢复到 root namespace
        pm.namespace(set=':')
    else:
        # merge 模式：用 file -import 命令
        time_range = "override" if fill_timeline else "combine"
        if namespace:
            cmd = f'file -import -type "FBX" -ignoreVersion -ra true -mergeNamespacesOnClash true -namespace ":{ns_name}" -options "fbx" -pr -importTimeRange "{time_range}" "{path}";'
        else:
            cmd = f'file -import -type "FBX" -ignoreVersion -ra true -options "fbx" -pr -importTimeRange "{time_range}" "{path}";'
        print(f"[import_fbx] Executing: {cmd}")
        pm.mel.eval(cmd)
    
    # 调试输出
    if ns_name:
        ns_joints = pm.ls(f"{ns_name}:*", type='joint')
        print(f"[import_fbx] Joints in namespace '{ns_name}': {len(ns_joints)}")
        if ns_joints:
            print(f"[import_fbx] Sample joints: {[str(j) for j in ns_joints[:5]]}")

def import_scene(
    path: str,
    *,
    namespace: str = ":",
    merge_namespaces_on_clash: bool = True,
    import_frame_rate: bool = False,
    import_time_range: Literal["combine", "override", "keep"] = "combine",
):
    """
    Import a Maya scene file (.ma/.mb) into the current scene.

    Args:
        path: Path to the Maya scene file to import.
        namespace: Namespace for imported objects. 
                   Default ":" imports into root namespace.
        merge_namespaces_on_clash: If True, merge namespaces when they clash.
                                   Default True (for root namespace behavior).
        import_frame_rate: If True, import the frame rate from the file.
                          Default False.
        import_time_range: How to handle time range. 
                          "combine" - combine with current range (default)
                          "override" - override current range
                          "keep" - keep current scene time range unchanged
    """
    # Determine file type from extension
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mb":
        file_type = "mayaBinary"
    elif ext == ".ma":
        file_type = "mayaAscii"
    else:
        raise ValueError(f"Unsupported file type: {ext}. Expected .ma or .mb")

    # Build the MEL command
    cmd_parts = [
        'file',
        '-import',
        f'-type "{file_type}"',
        '-ignoreVersion',
        '-ra true',
        f'-mergeNamespacesOnClash {"true" if merge_namespaces_on_clash else "false"}',
        f'-namespace "{namespace}"',
        '-options "v=0;"',
        '-pr',
    ]

    if import_frame_rate:
        cmd_parts.append('-importFrameRate true')

    if import_time_range != "keep":
        cmd_parts.append(f'-importTimeRange "{import_time_range}"')

    # Add the file path (with proper escaping)
    cmd_parts.append(f'"{path}"')

    cmd = ' '.join(cmd_parts)
    pm.mel.eval(cmd)

import pymel.core as pm
from .utils import mel_cmds
from maya import cmds


def create_hik_definition(name):
    mel_cmds('hikCreateDefinition;')
    old = mel_cmds('hikGetCurrentCharacter();')
    new = cmds.rename(old, name)
    mel_cmds(f'hikSetCurrentCharacter("{new}");')
    return new

def set_hik_bind(source_char, dest_char):
    """
    Set the source character to the destination character.
    """
    mel_cmds(f'hikSetCharacterInput("{dest_char}", "{source_char}");')
    return dest_char
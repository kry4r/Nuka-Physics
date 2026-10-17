# Sample editor extensions, one native and one Python, each adding a panel, an action and a
# viewport overlay. The build lays them out under src/extensions/, where the editor looks.
add_subdirectory(hello_cpp)
nuka_add_editor_extension_folder(hello_py)

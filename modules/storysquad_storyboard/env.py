import os

STORYBOARD_DEV_MODE = os.getenv("STORYBOARD_DEV_MODE") == "True"
STORYBOARD_RENDER_PATH = os.getenv("STORYBOARD_RENDER_PATH")
STORYBOARD_TMP_PATH = os.path.join(os.getenv("STORYBOARD_RENDER_PATH"), "tmp")
STORYBOARD_FFMPEG_PATH = os.getenv("STORYBOARD_FFMPEG_PATH")



def assure_os_perfect_path(os_path: str, override_os=None) -> str:
    """This function takes a string that is a path and returns a string that is a path that is perfect for the OS
        with and additional os.sep at the end of the path, uses os.sep and os.path.join
    >>> print(assure_os_perfect_path('C:/Users/JohnDoe/Desktop'))
    C:\\Users\\JohnDoe\\Desktop\\
    >>> print(assure_os_perfect_path('C:/Users/JohnDoe/Desktop/'))
    C:\\Users\\JohnDoe\\Desktop\\
    >>> pt = r'C:\\Users\\JohnDoe\\Desktop'
    >>> print(assure_os_perfect_path(pt))
    C:\\Users\\JohnDoe\\Desktop\\
    >>> pt = r'\\home\\JohnDoe\\Desktop'
    >>> print(assure_os_perfect_path(pt))
    /home/JohnDoe/Desktop/
    """
    # Check if the OS is Windows or Linux based
    if ':' in os_path:  # Windows
        path_type = "win"
    if '/' in os_path and ':' in os_path:  # invalid
        # convert to win path
        os_path = os_path.replace("/", "\\")
        # correct path specifier
        # os_path = os_path.replace(":", ":\\")
        path_type = "win"
    elif '/' in os_path:  # Linux
        path_type = "linux"
    elif '~' in os_path:  # Linux
        path_type = "linux"

    if os_path[0] == "\\":  # windows path sep with no drive letter = linux path
        path_type = "linux"
        os_path = os_path.replace("\\", "/")

    if override_os == "linux":
        path_type = "linux"
    elif override_os == "windows":
        path_type = "win"

    if path_type == "win":
        # correct path specifier
        os_path = os_path.replace("/", "\\")
        if os_path[-1] != "\\":
            os_path += "\\"
    elif path_type == "linux":
        os_path = os_path.replace("\\", "/")
        if os_path[-1] != "/":
            os_path += "/"

    return os_path


STORYBOARD_RENDER_PATH = assure_os_perfect_path(STORYBOARD_RENDER_PATH)
STORYBOARD_TMP_PATH = assure_os_perfect_path(STORYBOARD_TMP_PATH)
STORYBOARD_FFMPEG_PATH = STORYBOARD_FFMPEG_PATH
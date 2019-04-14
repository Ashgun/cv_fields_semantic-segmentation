def GetImageSize():
    return 256

def GetChannelsNum():
    return 3;

def GetClasses():
    return ['resources']

def GetClassesCount():
    return len(GetClasses())

def GetModelName():
    return 'ui_elements_detection_' + str(GetImageSize())

def GetNumIteration():
    return 6

def GetDataDirectory():
    return './'

source_data_path = GetDataDirectory() + '/src_imgs/'
training_data_path = GetDataDirectory() + '/training/'
validation_data_path = GetDataDirectory() + '/validation/'

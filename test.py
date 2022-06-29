#Create null object
TheNCE = TheSystem.NCE
for k in range(9, 177):
    TheNCE.AddObject()
for i in range(9, 177):
    locals()[f'object_{i}'] = TheNCE.GetObjectAt(i)
for i in range(9, 177):
    locals()[f'oType_{i}'] = locals()[f'object_{i}'].GetObjectTypeSettings(ZOSAPI.Editors.NCE.ObjectType.CADPartSTEPIGESSAT)
#Import CAD files
##Test IF can find the CAD file
if os.path.isfile(TheApplication.ObjectsDir + "\\CAD Files\\1.igs"):
    oType_1.FileName1 = '1.igs'
else:
    raise ImportError("CAD file not found")

if os.path.isfile(TheApplication.ObjectsDir + "\\CAD Files\\2.igs"):
    oType_2.FileName1 = '2.igs'
else:
    raise ImportError("CAD file not found")

object_1.ChangeType(oType_1)

##LOOP the index to import all CAD file
for i in range(1, 169):
    if os.path.isfile(TheApplication.ObjectsDir + "\\CAD Files\\{}.igs".format(i)):
        locals()[f'oType_{i+10}'].FileName1 = '{}.igs'.format(i)
        locals()[f'object_{i+10}'].ChangeType(locals()[f'oType_{i+10}'])
    else:
        raise ImportError("CAD file not found")

for i in range(9, 177):
    locals()[f'cad{i}_data'] = locals()[f'object_{i}'].ObjectData
    locals()[f'cad{i}_data'].set_Scale(2)

for i in range(9, 177):
    locals()[f'cad{i}_typedata'] = locals()[f'object_{i}'].TypeData
    locals()[f'cad{i}_typedata'].ObjectIsADetector = 1


#shade model
analysis = TheSystem.Analyses.New_Analysis(ZOSAPI.Analysis.AnalysisIDM.NSCShadedModel);
analysis.WaitForCompletion();




for i in range(9, 177):
    locals()[f'object_{i}'].TiltAboutX = 180


# Ray tracing
NSCRayTrace = TheSystem.Tools.OpenNSCRayTrace()
NSCRayTrace.SplitNSCRays = True
NSCRayTrace.ScatterNSCRays = True
NSCRayTrace.UsePolarization = True
NSCRayTrace.IgnoreErrors = True
NSCRayTrace.SaveRays = False

NSCRayTrace.Run()
# ! [e02s02_py]

lastValue = []
lastValue.append(0)
print('Beginning ray trace:')
while NSCRayTrace.IsRunning:
    currentValue = NSCRayTrace.Progress
    if currentValue % 2 == 0:
        if lastValue[len(lastValue) - 1] != currentValue:
            lastValue.append(currentValue)
            print(currentValue)
NSCRayTrace.WaitForCompletion()
NSCRayTrace.Close()


# open detector
detector = TheSystem.Analyses.New_Analysis(ZOSAPI.Analysis.AnalysisIDM.DetectorViewer)
detector.ApplyAndWaitForCompletion

result = []

# Collect lumen from each CAD part
for i in range(9, 178):
    flux_bool_return,total_flux = TheNCE.GetDetectorData(i, 0, 0, 0)
    result.append(total_flux)


import numpy as np

np.savetxt("resultee.csv", result, delimiter=',', )

#plot
import matplotlib.pyplot as plt
x = np.linspace(1, 168, 168)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, result)
plt.show()


# save the lumen data
data = np.loadtxt("result.csv", delimiter = ',')
data = np.column_stack((data, result))
np.savetxt('result.csv', data, delimiter= ',')

# save the input array
input_array = np.loadtxt("input.csv", delimiter = ',')
input_array = np.column_stack((input_array, input))
np.savetxt('input.csv', input_array, delimiter= ',')
# set opacity
for i in range(9, 177):
    locals()[f'cad{i}_drawdata'] = locals()[f'object_{i}'].DrawData
    locals()[f'cad{i}_drawdata'].Opacity = ZOSAPI.Common.ZemaxOpacity.P30


# enable blocks
blocks_list_1shape = [1, 3, 4, 10, 11, 12, 16, 17, 18, 19, 20, 21,
               22, 23, 39, 49, 73, 90, 97, 98, 126, 127]
blocks_list_2shape = [2, 3, 4, 9, 13, 14, 15, 19, 21, 25,
                26, 29, 30, 31, 35, 51, 75, 84, 85, 86, 93, 96]
blocks_list_3shape = [1, 3, 4, 5, 10, 11, 12, 16, 29, 30,
                31, 35, 36, 42, 54, 55, 56, 68, 69, 71, 72, 73,
                74, 78, 79, 80, 90, 91, 92, 93, 96, 127]


blocks_list_4shape = [2, 3, 4, 5, 6, 7, 9, 11, 13, 14, 15, 16, 19,
                20, 21, 26, 50, 51, 52, 54, 55, 56, 60, 66, 67, 70, 71,
                72, 74, 75, 79, 99, 100, 105, 106, 112, 123, 124, 125, 129, 130, 132,
                133, 134, 135, 136, 137, 138, 140, 143, 144, 145, 146]

blocks_list_5shape = [8, 11, 13, 15, 16, 20, 22, 23, 32, 39,56, 57, 58, 59, 60, 66, 67,
                      68, 69, 70, 71, 76, 77, 78, 80, 85, 86, 89, 90, 91, 92,
                      93, 94, 95, 96, 97, 98, 99, 121, 122, 123,
                      124, 125, 126, 128, 155]

blocks_list_0shape = [5, 29, 30, 31, 35, 36, 41, 42, 47, 54, 55, 56, 57, 61, 62, 63,
                      66, 67, 68, 70, 78, 79, 80, 84, 85, 86, 93, 96, 100, 101, 106,
                      108, 109, 120, 121, 122, 123, 124]

blocks_list_6shape = [8, 11, 13, 15, 16, 19, 20, 24, 25, 26, 27, 28, 37, 38, 40, 73, 74, 75, 76,
                      77, 90, 91, 92, 104, 105, 106, 107, 108, 127]

blocks_list_7shape = [4, 14, 15, 16, 18, 21, 28, 81, 82, 83, 86, 87, 92, 93, 114, 115, 116, 117,
                      118, 119, 159, 167, 168]

blocks_list_8shape = [1, 3, 4, 5, 6, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                      26, 29, 30, 32, 39, 54, 55, 56, 57, 61, 62, 63, 73, 85, 86, 87, 88, 89, 90,
                      94, 95, 97, 98, 99, 100, 101, 126, 127, 128]

blocks_list_9shape = [1, 2, 3, 4, 5, 6, 8, 11, 13, 15, 16, 17, 18, 20, 22, 23, 27, 28, 30, 31, 39, 54,
                      55, 56, 57, 61, 62, 63, 97, 98, 99, 100, 101]

blocks_list_Ashape = [6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 26, 27, 28, 31, 51, 52, 53, 58, 59, 61,
                      69, 71, 72, 73, 74, 75, 76, 77, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
                      101, 102, 125, 126, 127, 128, 149, 150, 154, 155, 156, 157, 160, 164]

blocks_list_Bshape = [1, 2, 3, 4, 9, 13, 14, 15, 19, 21, 25, 26, 29, 30, 31, 51, 52, 53,
                      54, 55, 68, 69, 71, 72, 73, 74, 75, 80, 85, 86, 89, 93, 96, 125, 126, 128, 129, 130,
                      131, 135]

blocks_list_Cshape = [68, 69, 71, 72, 80, 85, 86, 87, 88, 89, 125, 126, 128, 130, 131, 134, 140, 145, 146,
                      156, 157, 160]

blocks_list_Dshape = [2, 3, 4, 19, 21, 25, 26, 29, 34, 35, 40, 41, 42, 43, 44, 56, 57, 58, 59, 60, 67, 68,
                      69, 75, 76, 98, 99, 107, 133, 134, 136, 137, 140, 141, 143, 144, 151]

blocks_list_lowestbound = [48, 64, 65, 81, 102, 103, 104, 105, 110, 111, 112, 113, 114, 115, 116, 117, 118,
                           119, 132, 133, 134, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 152,
                           153, 154, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168]

blocks_list_lowestbound_top1 = [45, 46, 47, 61, 62, 63, 66, 67, 70, 82, 83, 84, 100, 101, 106, 108, 109, 120,
                                121, 122, 123, 124, 130, 131, 136, 137, 150, 151, 156, 157]

blocks_list_lowestbound_top2 = [5, 32, 33, 34, 40, 41, 42, 43, 44, 54, 55, 56, 57, 58, 59, 60, 68, 69, 71, 72,
                                78, 79, 80, 85, 86, 87, 88, 89, 94, 95, 98, 99, 107, 125, 126, 128, 129, 135, 149, 155]

blocks_list_lowestbound_top3 = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                                27, 28, 29, 30, 31, 50, 51, 52, 53, 73, 74, 75, 76, 77, 78]

blocks_onepart = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
              62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 113, 114, 115, 116, 117, 118,
              119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
              140, 141, 142, 143, 144, 145, 146, 147, 148]

blocks_secondpart = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                     45, 46, 47, 48, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
                     103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
                     161, 162, 163, 164, 165, 166, 167, 168]

blocks_list_all = [i for i in range(1, 169)]

size = len(blocks_list_lowestbound_top1)

input = []
for _ in range(168):
    input.append(0)
for i in blocks_secondpart:
    input[i-1] = 1



TheNCE = TheSystem.NCE
for k in range(4):
    TheNCE.AddObject()

for i in range(177, 177+size):
    locals()[f'object_{i}'] = TheNCE.GetObjectAt(i)
for i in range(177, 177+size):
    locals()[f'oType_{i}'] = locals()[f'object_{i}'].GetObjectTypeSettings\
        (ZOSAPI.Editors.NCE.ObjectType.CADPartSTEPIGESSAT)

for i in range(size):
    if os.path.isfile(TheApplication.ObjectsDir + "\\CAD Files\\{}.igs".format(blocks_list_lowestbound_top1[i])):
        locals()[f'oType_{177+i}'].FileName1 = '{}.igs'.format(blocks_list_lowestbound_top1[i])
        locals()[f'object_{177+i}'].ChangeType(locals()[f'oType_{177+i}'])
    else:
        raise ImportError("CAD file not found")

for i in range(177, 177+size):
    locals()[f'object_{i}'].TiltAboutX = 180

# Can't set particles value striaght, best approach is to duplicate existed rows and append CAD
for i in range(177, 199):
    locals()[f'object_{i}'] = TheNCE.GetObjectAt(i)
    locals()[f'object_{i}'].VolumePhysicsData.Model = ZOSAPI.Editors.NCE.VolumePhysicsModelType.DLLDefinedScattering
    locals()[f'object_{i}'].VolumePhysicsData.ModelSettings.DLL = str('Mie.dll')
    locals()[f'object_{i}'].VolumePhysicsData.ModelSettings.MeanPath = 100
    locals()[f'object_{i}'].VolumePhysicsData.ModelSettings.ParticleIndex = 1.6
    locals()[f'object_{i}'].VolumePhysicsData.ModelSettings.ParticleDensity = 1734100000



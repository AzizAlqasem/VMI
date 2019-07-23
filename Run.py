import numpy as np
import matplotlib.pyplot as plt
#Local Import
import vmi


__version__ = '0.1.0'


data = np.load('data.npy')
psf = np.loadtxt('psfd3.txt')

v = vmi.VMI(data)

#1- Deconvolote 
v.deconv(psf, iter=5)

#2- Rotate the image
v.rotate(60)

#3- Center $ Croup the Image
v.crop()

#4- Truncate the Image
v.truncate()

#5- Darkspots filter
v.darkspot_filter()

#6- Avaerge qurters
v.get_image_quadrants(symmetry_axis=(0,1), use_quadrants=[True, True, True, False]) # Here we average along the first 3 quadrants
v.put_image_quadrants(v.quadrants, symmetry_axis=(0,1))

#______________    Not working  ___________ 
###7- Correct the Image distortion (make sure that the image is centered)
###v.correct_image_distortion(method='argmax', radial_range = [0, 500])
#______________       End       ___________

#7- Abel transform
v.basex_transform()

#8- Get velocities at 15 degrees step: (0, 15, 30, ..., 90)
v.to_polar()
v.electron_count()

#9- Plot Results
f, ax = plt.subplots(2,1, dpi = 200)

ax[0].set_title('Final')
ax[0].imshow(v.data, cmap = 'gray', vmin = 10, vmax = 20000)

v.plot_electron_count_vs_energy()
ax[1].set_title('Electron Energy at different angles')
ax[1].set_xlabel('Energy')
ax[1].set_ylabel('Count')

plt.show()





"""
pa = np.arange(0, np.pi, 100)
lo = [0,2]
#r  = v2.lin_basex_transform(proj_angles=pa, legendre_orders= lo)
fig, a = plt.subplots(1,2, dpi = 200)
a[0].imshow(r.transform, cmap = 'gray', vmin = 100, vmax = 20000)
radial = r.radial
speed = r.Beta[0][::-1]
#speed /= speed[200:].max()
a[1].plot(radial , speed)
a[1].set_xlim(300,900)
a[1].set_ylim(-0.25,1)
plt.show()

_________

#v.get_electron_count2(dt = 5)
#lb = v.lin_basex_transform(proj_angles=[np.pi/4])
fig, a = plt.subplots(4,2)
a = a.flatten()
ang = [0, 15, 30, 45, 60, 75, 90][::-1]
for i,l in enumerate(v.lines.T):
    a[i].plot(l[10:500], label = ang[i])
    a[i].legend()
a[-1].plot(lb.Beta[0][10:500], label = 'lb - 45')
a[-1].legend()
plt.show()
"""
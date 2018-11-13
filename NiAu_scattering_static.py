# coding: utf8
from lammps import lammps, PyLammps
lmp = lammps()
L = PyLammps(ptr=lmp)
import time
from mpi4py import MPI
import scipy.io
import numpy as np
from numpy import linalg as LA
import math, random
from shutil import copyfile

# Function definitions:
def getcoords_old():	# slow, but doesn't need additional lammps python interface
	nPart1 = 0
	nPart2 = 0
	nPartsub = 0

	coords1 = []
	coords2 = []
	coordssub = []
	indices1 = []
	indices2 = []
	indicessub = []
	print("get coords...")
	startt = time.time()
	for i in range(L.system.natoms):
		atype = L.atoms[i].type
		if atype == 1:
			nPart1 += 1
			#coords1 = coords1 + L.atoms[i].position	# wenns tuples sind
			coords1.append(L.atoms[i].position)
			indices1.append(i)
		elif atype == 2:
			nPart2 += 1
			coords2.append(L.atoms[i].position)
			indices2.append(i)
		else:
			nPartsub += 1
			coordssub.append(L.atoms[i].position)
			indicessub.append(i)
	
	print("elapsed time: {}".format(time.time()-startt))
	
	coords1 = np.array(coords1)	# convert list to array
	coords2 = np.array(coords2)
	coordssub = np.array(coordssub)
	return nPart1, nPart2, nPartsub, coords1, coords2, coordssub, indices1,indices2,indicessub

def getcoords():	# fast
	nPart1 = 0
	nPart2 = 0
	nPartsub = 0

	coords1 = []
	coords2 = []
	coordssub = []
	indices1 = []
	indices2 = []
	indicessub = []
	#print("get coords...")
	#startt = time.time()
	
	successflag = False
	trials = 0
	while successflag == False:
		# Wenn sich die Zahl der atome ändert dann klappt diese methode nicht mehr und wir springen zu einer langsameren alternativen Methode
		# (WARNING: Library error in lammps_gather_atoms (../library.cpp:822) und WARNING: Library error in lammps_gather_atoms (../library.cpp:822))

		x = np.array(lmp.gather_atoms("x",1,3)) 	# gather atoms gives per-atom data ordered by atom ID
		x = np.reshape(x,(L.system.natoms,3))
		#types = lmp.extract_atom("type",0)	# get types
		types = np.array(lmp.gather_atoms("type",0,1))
			
		for i in range(L.system.natoms):
			atype = types[i]
			if atype == 1:
				nPart1 += 1
				coords1.append(x[i])
				indices1.append(i)
			elif atype == 2:
				nPart2 += 1
				coords2.append(x[i])
				indices2.append(i)
			else:
				nPartsub += 1
				coordssub.append(x[i])
				indicessub.append(i)
		
		if nPart1 > 0 and nPart2 > 0 and nPartsub > 0:
			successflag = True
		else:
			trials=trials+1
		
		if trials > 1:
			(nPart1, nPart2, nPartsub, coords1, coords2, coordssub, indices1,indices2,indicessub) = getcoords_old()	# alternative methode verwenden
			break
			
	#print("elapsed time: {}".format(time.time()-startt))
	
	coords1 = np.array(coords1)	# convert list to array
	coords2 = np.array(coords2)
	coordssub = np.array(coordssub)
	return nPart1, nPart2, nPartsub, coords1, coords2, coordssub, indices1,indices2,indicessub
	
	
def resetcoords(coords1,coords2,indices1,indices2):

	l=0
	
	startt = time.time()
	for i in indices1:
		L.atoms[i].position = coords1[l]
		l=l+1
	
	l=0
	for i in indices2:
		L.atoms[i].position = coords2[l]
		l=l+1
		
	print("elapsed time: {}".format(time.time()-startt))
	return
	
def calcscatteringvector(theta,m,Eel):
	
	# DEFINE CONSTANTS:
	elch = 1.602176565E-19	# C
	c = 299792458			# speed of light m/s
	mel = 9.10938291E-31
	u = 1.660538E-27		# unified atomic mass unit in kg
	Eel0 = mel*c*c

	Et = 2*Eel*(Eel+2*Eel0)/(c**2*m) * math.sin(theta/2)**2 		# transferred Energy (electron to atom) in J
	print("Transferred energy: {} eV at theta = {} degree".format(Et/elch,theta/math.pi*180))

	ve = c*(1-Eel0**2/(Eel0+Eel-Et)**2)**0.5	# speed of electron after scattering in LJ-units
	ve0 = c*(1-Eel0**2/(Eel0+Eel)**2)**0.5	# speed of electron before scattering

	vatom = (2*Et/m)**0.5 				# speed of hit atom after scattering 
		
	psi = math.asin(mel*ve/(1-(ve/c)**2)**0.5/(m*vatom)*math.sin(theta))	# from momentum conservation (p_el*sin(theta) = p_atom *sin(psi) & relativistic p_el and classic pa=m*vatom
	phi = random.uniform(0,1)*2*math.pi

	v_sc = vatom * np.array([math.cos(phi)*math.sin(psi), math.sin(phi)*math.sin(psi), math.cos(psi)])
	return v_sc, Et
	
def thermalize(Temp):
	nstep = 500
	while L.eval("temp") > Temp+Temp*0.05:
		initTemp = L.eval("temp")
		L.run(nstep)
		print("Temp in thermalize function: {} (start: {}, goal: {})".format(L.eval("temp"),initTemp,Temp+Temp*0.05))
	return

def calccoordNN(coords1,coords2, NNthresh):

	nPart1 = coords1.shape[0]
	nPart2 = coords2.shape[0]

	countNN11 = np.zeros(nPart1)	# Au NN aus Sicht der Au Atome
	countNN22 = np.zeros(nPart2)	# Ni NN aus Sicht der Ni Atome
	countNN21 = np.zeros(nPart2)	# Au NN aus Sicht der Ni Atome
	countNN12 = np.zeros(nPart1)	# Ni NN aus Sicht der Au Atome
	
	maxd2 = NNthresh**2;
	
	# Loop over all particle pairs for element 1:
	for partA in range(0,nPart1-1):
		for partB in range(partA+1,nPart1):
			# Calculate particle-particle distance
			dr = coords1[partA,:] - coords1[partB,:]
			
			# Fix according to periodic boundary conditions
			# dr = distPBC3D(dr,L);
			
			dr2 = np.dot(dr,dr) # = dr[1].*dr[1]+dr[2].*dr[2]+dr[3].*dr[3]
			if dr2<maxd2:
					countNN11[partA] = countNN11[partA] + 1
					countNN11[partB] = countNN11[partB] + 1
					
	# Loop over all particle pairs for element 2:
	for partA in range(0,nPart2-1):
		for partB in range(partA+1,nPart2):
			# Calculate particle-particle distance
			dr = coords2[partA,:] - coords2[partB,:]
			# Fix according to periodic boundary conditions
			# dr = distPBC3D(dr,L)
			
			dr2 = np.dot(dr,dr) # = dr[1].*dr[1]+dr[2].*dr[2]+dr[3].*dr[3]

			if dr2<maxd2:
				countNN22[partA] = countNN22[partA] + 1
				countNN22[partB] = countNN22[partB] + 1
	
	# Loop over all particle pairs between element1 and element 2:
	for partA in range(0,nPart1-1):
		for partB in range(0,nPart2-1):
			# Calculate particle-particle distance
			dr = coords1[partA,:] - coords2[partB,:]
			# Fix according to periodic boundary conditions
			# dr = distPBC3D(dr,L)
			
			dr2 = np.dot(dr,dr) # = dr[1].*dr[1]+dr[2].*dr[2]+dr[3].*dr[3]

			if dr2<maxd2:
				countNN12[partA] = countNN12[partA] + 1
				countNN21[partB] = countNN21[partB] + 1
	
	return countNN11, countNN12, countNN21, countNN22
					
# important parameters:
Temp = 300.0
xNi = 0.3
dCluster = 20 #40
Lsub = 40 #70
maxnel = 1000000;

# if larger than 0 a full xyz time series is written to a file every fullsavedata-ths step
fullsavedata = 0

# time spans for scattering simulation in ps:
initequalt = 1					# initial equilibration time between velocity reset and new scattering event
dcheckafterscatteringt = 0.4	# time between scattering event and displacement check
equilibrationt = 0.5			# time for equilibration
# continue from existing geometry?
continueprevsim = False

#reset geometry after every run?
resetgeom = True
selectfromNIST = False

# MD PARAMETERS:
L.units("metal")
L.atom_style("atomic")
L.atom_modify("map array")

latticeconst = 4.08
dispthresh = (4.08+3.52)/2/math.sqrt(2)*0.90
NNthresh = (4.08+3.52)/2/math.sqrt(2)*1.2

L.timestep(0.0005) # time-step in ps (metal units)
dsub = 10

density = 2000
Amass = 12.011*1.660538E-27
nPartsub = int(round(Lsub*Lsub*dsub*1e-30*density/Amass))
print(nPartsub)

#Au-C: (Lewis.2000)
sigAuC = 2.74	# in A
epsAuC = 0.022	# in eV

#Ni-C: (Huang.2003, Ryu2010)
sigNiC = 2.852
epsNiC = 0.023049

# DEFINE CONSTANTS:
elch = 1.602176565E-19	# C
c = 299792458			# speed of light m/s
mel = 9.10938291E-31
u = 1.660538E-27		# unified atomic mass unit in kg
Eel0 = mel*c*c
a0 = 0.52917721067e-10	# Bohrscher radius

mC = 12.011*u
mAu = 196.96655*u
mNi = 58.6934*u

#experimental parameters:
HT = 300;						# Primary electron energy in keV
Eel = HT * 1E3 * elch;			# energy of incoming electrons in J
Et = 0.0;
Etmax1 = 2*Eel*(Eel+2*Eel0)/(c**2*mAu);
Etmax2 = 2*Eel*(Eel+2*Eel0)/(c**2*mNi);
Etmin = 0.5*elch;

# load NIST scattering cross sections from file:
scatterfile = scipy.io.loadmat('Au300.mat', squeeze_me=True, struct_as_record=False)
scattercross1 = np.array(scatterfile['cross_section'])
scatterangle1 = np.array(scatterfile['deg'])

scatterfile = scipy.io.loadmat('Ni300.mat', squeeze_me=True, struct_as_record=False)
scattercross2 = np.array(scatterfile['cross_section'])
scatterangle2 = np.array(scatterfile['deg'])

# generate cumulative distribution function considering given angular ranges
# set theta range:

thetamin = 2*math.asin(math.sqrt(Etmin/Etmax1))*180/math.pi;
thetamax = 180;

X1 = scatterangle1[scatterangle1>=thetamin]
X1 = X1[X1<=thetamax]
indmin = np.where(scatterangle1==min(X1))
indmin = np.asarray(indmin).item()
indmax = np.where(scatterangle1==max(X1))
indmax = np.asarray(indmax).item()
cdf1 = np.cumsum(scattercross1[indmin:indmax])

# crosstot1 = 2*math.pi*np.trapz(scatterangle1/180*math.pi,scattercross1*math.sin(scatterangle1/180*math.pi))*a0**2	# trapz performs a numerical integration after the trapezoidal rule
# crosssec1 = 2*math.pi*np.trapz(X1/180*math.pi,P*math.sin(scatterangle1[indmin:indmax]/180*math.pi))*a0**2
# scalingfactor1 = crosstot1/crosssec1

# generate cumulative distribution function considering given angular ranges
# set theta range:

thetamin = 2*math.asin(math.sqrt(Etmin/Etmax2))*180/math.pi;
thetamax = 180;

X2 = scatterangle2[scatterangle2>=thetamin]
X2 = X2[X2<=thetamax]
indmin = np.where(scatterangle2==min(X2))
indmin = np.asarray(indmin).item()
indmax = np.where(scatterangle2==max(X2))
indmax = np.asarray(indmax).item()
cdf2 = np.cumsum(scattercross2[indmin:indmax])

if continueprevsim == False:
	L.boundary("p","p","f") # "f" non-periodic and fixed

	L.region("cluster","block",(Lsub-dCluster)/2,Lsub-(Lsub-dCluster)/2,(Lsub-dCluster)/2,Lsub-(Lsub-dCluster)/2,dsub,dsub+dCluster)
	L.region("subst","block",0,Lsub,0,Lsub,2,dsub)
	L.region("substinteg","block",0,Lsub,0,Lsub,dsub-4,dsub)
	L.region("simregion","block",0,Lsub,0,Lsub,0,dCluster*3)
	L.region("allintegr","union",2,"substinteg","cluster")

	L.lattice("fcc",latticeconst)

	L.create_box(3,"simregion")	# create box containing 3 elements over full simulation region

	#L.mass(1,196.96655) 	# molecular weight in g/mol  Au
	#L.mass(2,58.6934) 		# Ni
	L.mass(3,12.011) 		# C

	L.create_atoms(1,"region","cluster")	# fill cluster region with atoms of type 1
	L.create_atoms(3,"random",nPartsub,4723738,"subst")	# fill substrate region with atoms of type 3

	L.group("substinteggr","region","substinteg")
	L.group("clusterregiongr","region","cluster") 

	L.group("allintegrategr","region","allintegr") 

	L.set("region","cluster","type/fraction",2,xNi,1234) 
	L.group("clustergr","type",1,2)

	# mit Zhou-Potential:
	L.pair_style("hybrid","eam/alloy","lj/cut",10.0,"tersoff")
	L.pair_coeff("* * eam/alloy NiAu_Zhou.eam.alloy Au Ni NULL")
	L.pair_coeff(1,3,"lj/cut",epsAuC,sigAuC,10.0)
	L.pair_coeff(2,3,"lj/cut",epsNiC,sigNiC,10.0)
	L.pair_coeff("* *","tersoff","SiC.tersoff","NULL NULL C")

	# mit Ralf Meyers Potential:
	# L.pair_style("hybrid","eam/fs","lj/cut",10.0,"tersoff")
	# L.pair_coeff("* * eam/fs Ni_Au_SMATB.eam.fs Au Ni NULL")
	# L.pair_coeff(1,3,"lj/cut",epsAuC,sigAuC,10.0)
	# L.pair_coeff(2,3,"lj/cut",epsNiC,sigNiC,10.0)
	# L.pair_coeff("* *","tersoff","SiC.tersoff","NULL NULL C")

	# L.pair_style("hybrid eam lj/cut 10.0 tersoff")
	# L.pair_coeff("* * eam Au_u3.eam")
	# L.pair_coeff(1,2,"lj/cut",epsAuC,sigAuC,10.0)
	# L.pair_coeff("* * tersoff SiC.tersoff NULL C")

	(nPart1, nPart2, nPartsub, coords1, coords2, coordssub,indices1,indices2,indicessub) = getcoords()
	print("no. of atoms in the system: {} Au, {} Ni, {} C".format(nPart1,nPart2,nPartsub))
	
	scaleF = nPart1*cdf1[-1]/(nPart2*cdf2[-1])
	print(scaleF)

	L.minimize(1e-6,1e-7,5000,50000)
	print("initial minimization completed.")

	L.compute("K","all","ke/atom")
	L.compute("P","all","pe/atom")
	L.compute("coordno","all","coord/atom","cutoff",NNthresh)	# calculate coordination no between all atoms of type 1 or 2 (exclude substrate atoms (type 3)

	#L.dump("d1","all","xyz",1000,"output.xyz")
	#L.dump("d1","all","custom",1000,"output.out", "id", "type","x","y","z","vx","vy","vz","c_K","c_P","c_coordno")
	#L.dump("d1","xtc","atom",100,"T" + str(Temp) + "_Fulltrajectories.xtc",100)	# last value gives precision, a value of 100 means that coordinates are stored to 1/100 nanometer accuracy
	#L.dump_modify("d1","element","Au","Ni","C")
	#L.dump_modify("d1","pbc","yes") # remap atoms via periodic boundary conditions
	#L.dump("trajectory","all","atom",100, "T" + str(Temp) + "_Fulltrajectories.lammpstrj")

	# Set thermo output to log file
	# L.thermo_style("custom","step","atoms","temp","etotal","pe","ke","dt")  
	# L.thermo(100)

	L.thermo_modify("lost","ignore","flush","yes")
	L.velocity("all","create",Temp,12345,"dist","gaussian","mom","yes","rot","yes")

	#L.fix("thermofix","all","temp/csvr",Temp,Temp,0.1,12345)	# does not perform time integration !
	L.fix("thermofix1","allintegrategr","langevin",1,1,0.1, 699483,"zero","yes")
	#L.fix("nveintegration","allintegrategr","nve/limit",0.05)	# performs time integration with modified velocities in micro-canonical ensemble
	L.fix("nveintegration","allintegrategr","nve")

	# initial equilibration:
	starttime = time.time()
	L.run(20000)
	print("dt at start: {} fs".format(L.eval("dt")*1000))
	print("T at start: {} K".format(L.eval("temp")))
	print("elapsed time: {} s".format(time.time() - starttime))

	L.unfix("nveintegration")	# remove time integration fix
	L.unfix("thermofix1")
	
	# Test Atom selection:
	# atom = 1
	# current_atomID = L.atoms[atom].id
	# current_atom_handle = L.atoms[atom]
	# L.group("selected","id","value",current_atomID)
	# L.velocity("selected","set",newv[0],newv[1],newv[2],"units","box")
		
	L.fix("thermofix","clustergr","langevin",500,500,0.1, 12345,"zero","yes") # does not perform time integration !
	L.fix("nveintegration","clustergr","nve")	# move only cluster atoms
	L.timestep(0.002)
	starttime = time.time()
	L.run(50000)
	print("dt at start: {} fs".format(L.eval("dt")*1000))
	print("T at start: {} K".format(L.eval("temp")))
	print("elapsed time: {} s".format(time.time() - starttime))

	L.unfix("thermofix")
	L.fix("thermofix2","clustergr","langevin",Temp,Temp,0.1, 12345,"zero","yes")
	L.fix("nveintegration","clustergr","nve")	# move only cluster atoms and surface substrate atoms

	starttime = time.time()
	L.run(20000)
	print("dt at start: {} fs".format(L.eval("dt")*1000))
	print("T at start: {} K".format(L.eval("temp")))
	print("elapsed time: {} s".format(time.time() - starttime))
	L.unfix("thermofix2")

	#initial equilibration procedure finished:
	print("****************************************")
	print("initial equilibration procedure finished, writing restart file")
	print("****************************************")

	L.write_restart("restart.equil")
else:
	L.read_restart("restart.equil")

L.velocity("all","create",Temp,12345,"dist","gaussian","mom","yes","rot","yes")
#L.fix("mainvartimestep","clustergr","dt/reset",10,"NULL","NULL",latticeconst/30,"emax",0.1,"units","box")
L.fix("mainvartimestep","clustergr","dt/reset",1,"NULL","NULL",0.1,"units","box")
L.fix("linmomfix","clustergr","momentum",1,"linear",1, 1, 1)	# remove linear momentum for better displacement detection

#L.fix("thermofix3","clustergr","langevin",Temp,Temp,0.05, 2614,"zero","yes")
#L.fix("thermofix3","clustergr","temp/csvr",Temp,Temp,0.1,12345)

nodispl = 0
nosputt = 0
displacement = False

# save starting coordinates:
(startnPart1, startnPart2, nPartsub, startcoords1, startcoords2, coordssub,startindices1,startindices2,indicessub) = getcoords()
startcoords = np.append(startcoords1,startcoords2,axis=0)

# Output starting geometry:
L.dump("Startgeom","all","custom",1, "Startgeom_Full" + ".xyz","id", "type","x","y","z","vx","vy","vz","c_K","c_P","c_coordno")
L.dump_modify("Startgeom","first","yes","pad",10)	# Schreib' beim nullten Schritt ein File raus
L.dump_modify("Startgeom","element","Au","Ni","C")
L.dump_modify("Startgeom","pbc","yes") # remap atoms via periodic boundary conditions
L.run(0)
L.undump("Startgeom")

L.dump("Startgeom","all","xyz",1, "Startgeom" + ".xyz")
L.dump_modify("Startgeom","first","yes","pad",10)	# Schreib' beim nullten Schritt ein File raus
L.dump_modify("Startgeom","element","Au","Ni","C")
L.dump_modify("Startgeom","pbc","yes") # remap atoms via periodic boundary conditions
L.run(0)
L.undump("Startgeom")

shotelectrons_temp = np.zeros((8, maxnel))
sputteringevents = np.zeros((6, maxnel))

for nel in range(0,maxnel):

	starttime = time.time()

	newv = [0,0,0]
	initialcoords = []
	clustercoords = []
	displacement = False
	
	if fullsavedata > 0:
		L.dump("savefullstep","all","custom",fullsavedata,"T" + str(Temp) + "eln_" + str(nel) + "_Full.xyz", "id", "type","x","y","z","vx","vy","vz","c_K","c_P","c_coordno")
		#L.dump("savefullstep","all","xyz",fullsavedata,"T" + str(Temp) + "elno_" + str(nel) + "_Full.xyz")
		L.dump_modify("savefullstep","element","Au","Ni","C")
	
	# Neue Geschwindigkeiten zuweisen:
	# L.unfix("nveintegration")		# stop integration to reasign group (in case atoms got lost during previous scattering event)
	
	#redefine cluster group (in case atoms were removed)
	L.unfix("mainvartimestep")
	L.unfix("linmomfix")
	#L.unfix("thermofix3")				# turn of termostat
	L.unfix("nveintegration")
	L.group("clustergr","delete")
	L.group("allintegrategr","delete")
	L.group("clustergr","type",1,2)		# Select all Au and Ni atoms and add them to the cluster group
	L.group("allintegrategr","union","clustergr","substinteggr") # recreate group for integration
	# L.velocity("clustergr","create",Temp,12345,"dist","gaussian","loop","geom")	# recreate velocities
	L.fix("linmomfix","clustergr","momentum",1,"linear",1, 1, 1)	# remove linear momentum for better 
	L.fix("nveintegration","clustergr","nve")	# move only cluster atoms and surface substrate atoms
	
	L.fix("clusterinit","clustergr","temp/csvr",Temp,Temp,0.1,4565465)	#thermalize cluster before scattering
	L.run(300)
	L.unfix("clusterinit")	# no thermostat off before scattering
	
	#L.fix("mainvartimestep","clustergr","dt/reset",1,"NULL","NULL",latticeconst/30,"emax",0.1,"units","box")
	L.fix("mainvartimestep","clustergr","dt/reset",1,"NULL","NULL",0.1,"units","box")
	
	print("T after initialisation: {} K; dt after initialisation: {} fs".format(L.eval("temp"),L.eval("dt")*1000))
	print("potential energy: {} eV, kinetic energy: {}".format(L.eval("pe"),L.eval("ke")))
	
	print("next electron no: {}".format(nel))
	# store intial coordinates and particle numbers:
	(nPart1, nPart2, nPartsub, coords1, coords2, coordssub,indices1,indices2,indicessub) = getcoords()
	initialcoords = np.append(coords1,coords2,axis=0)
	(coordsNN11,coordsNN12,coordsNN21,coordsNN22) = calccoordNN(coords1,coords2, NNthresh)#
	
	print("no. of atoms in the system: {} Au, {} Ni, {} C".format(nPart1,nPart2,nPartsub))

	scaleF = nPart1*cdf1[-1]/(nPart2*cdf2[-1])
	# -------------------------
	#       SCATTERING
	# -------------------------	
	if selectfromNIST:
		#select material:
		if random.uniform(0,1) > 1/scaleF:
			atype = 1
		else:
			atype = 2
	
		# select corresponding atom:
		if atype == 1:
			atom = random.randint(0,nPart1-1)
			atomid = indices1[atom]
			#theta from NIST-file:
			theta = X1[sum(cdf1< random.uniform(0,1)*cdf1[-1]) + 1]	# Generating a non uniform discrete random variable from, scattering data
		elif atype == 2:
			atom = random.randint(0,nPart2-1)
			atomid = indices2[atom]
			#theta from NIST-file:
			theta = X2[sum(cdf2< random.uniform(0,1)*cdf2[-1]) + 1]	# Generating a non uniform discrete random variable from, scattering data
		
		print("selected atom no. {} of type {}".format(atom,L.atoms[atomid].type))
		theta = theta*math.pi/180
		#theta = math.pi		# max. energy
		#theta = 0.001
	
	else:	# uniform probability distribution:
		
		# select atom:
		if random.uniform(0,1) > np.float(nPart2)/np.float(nPart1+nPart2):
			atom = random.randint(0,nPart1-1)
			atomid = indices1[atom]
		else:
			atom = random.randint(0,nPart2-1)
			atomid = indices2[atom]
			
		theta = random.uniform(0,math.pi)	# uniformly distributed theta 
		atype = L.atoms[atomid].type
	
	print("selected atom no. {} of type {}".format(atom,atype))
		
	# get atom and velocity from LAMMPS:
	current_atomID = L.atoms[atomid].id
	current_atom_handle = L.atoms[atomid]
	v = current_atom_handle.velocity   # identical with: L.atoms[atomid].velocity

	#print("velocity before scattering: {}".format(v))

	# CALCULATE SCATTERING VECTOR:
	
	if atype == 1:
		(v_sc, E_t) = calcscatteringvector(theta,mAu,Eel)
	elif atype == 2:
		(v_sc, E_t) = calcscatteringvector(theta,mNi,Eel)
	
	v_sc = v_sc * 0.01  # 1 meter/second = 0.01 angstroms/picosecond
	v_sc[2] = -v_sc[2] # negative z-direction

	#------------------------------------------------------
	#print("calculated scattering vector: {}".format(v_sc))

	newv = np.array(v) + np.array(v_sc)
	#print("calculated velocity vector: {}".format(newv))

	# put new atom-velocity into MD calculation:
	L.group("selected","id",current_atomID)
	#L.group("selected","id","value",current_atomID)
	L.velocity("selected","set",newv[0],newv[1],newv[2],"units","box")
	#L.atoms[atomid].velocities = (newv[0],newv[1],newv[2])	# andere Variante, funkt aber nicht Befehl wird ignoriert ?
	L.group("selected","delete")
	
	
	shotelectrons_temp[0,nel] = atomid
	shotelectrons_temp[1,nel] = E_t
	shotelectrons_temp[4,nel] = theta
	shotelectrons_temp[5,nel] = atype
	
	if atype == 1:		# which atom chosen: Au
		shotelectrons_temp[6,nel] = coordsNN11[atom] # How many Au neighbours
		shotelectrons_temp[7,nel] = coordsNN12[atom] # How many Ni neighbours
	elif atype == 2:	# Ni
		shotelectrons_temp[6,nel] = coordsNN21[atom] # How many Au neighbours
		shotelectrons_temp[7,nel] = coordsNN22[atom] # How many Ni neighbours
	
	
	L.run(50)
	print("T after scattering: {} K; dt after scattering: {} fs".format(L.eval("temp"),L.eval("dt")*1000))
	print("potential energy: {} eV, kinetic energy: {}".format(L.eval("pe"),L.eval("ke")))
	
	L.run(int(round(dcheckafterscatteringt/L.eval("dt"))))

	(nPart1, nPart2, nPartsub, coords1, coords2, coordssub,indices1,indices2,indicessub) = getcoords()
	clustercoords = np.append(coords1,coords2,axis=0)
	
	#L.fix("thermofix3","clustergr","langevin",Temp,Temp,0.05, 12345,"zero","yes")
	#L.fix("thermofix3","clustergr","temp/csvr",Temp,Temp,10,3463)
		
	if len(initialcoords) == len(clustercoords):
	
		dr = initialcoords - clustercoords
		# Fix according to periodic boundary conditions
		# Distance vector should be in the range -hLx -> hLx and -hLy -> hLy
		# Therefore, we need to apply the following changes if it's not in this range: 
		# Calculate the half box size in each direction
		hL = Lsub/2
		for dim in range(0,2):
			for part in range(0,nPart1 + nPart2-1):
				if dr[part][dim] > hL:
					dr[part][dim] = dr[part][dim] - Lsub
				elif dr[part][dim] < -hL:
					dr[part][dim] = dr[part][dim] + Lsub
		
		dr = np.absolute(LA.norm(dr, axis=1))
		drsorted = np.sort(dr,axis=None)
		print("largest displacement found: {} A (threshold: {} A)".format(drsorted[-1],dispthresh))
		
		if drsorted[-1] > dispthresh:
		#if np.allclose(LA.norm(initialcoords, axis=1),LA.norm(clustercoords, axis=1),atol = dispthresh) != False:
			print("displacement detected, 2nd check:")
			L.run(50)	
			(nPart1, nPart2, nPartsub, coords1, coords2, coordssub,indices1,indices2,indicessub) = getcoords()
			clustercoords = np.append(coords1,coords2,axis=0)
			dr = initialcoords - clustercoords
			# Fix according to periodic boundary conditions:
			hL = Lsub/2
			for dim in range(0,2):
				for part in range(0,nPart1 + nPart2-1):
					if dr[part][dim] > hL:
						dr[part][dim] = dr[part][dim] - Lsub
					elif dr[part][dim] < -hL:
						dr[part][dim] = dr[part][dim] + Lsub
			
			dr = np.absolute(LA.norm(dr, axis=1))
			drsorted = np.sort(dr,axis=None)
			print("largest displacement found: {} A (threshold: {} A)".format(drsorted[-1],dispthresh))
			
			if drsorted[-1] > dispthresh:
				print("displacement confirmed, equilibration...")
				displacement = True
				nodispl += 1
			
				shotelectrons_temp[2,nel] = drsorted[-1]	
				#shotelectrons_temp[3,nel] = index;
				if resetgeom == False:
					L.run(400)
					#L.fix("thermalize","clustergr","temp/csvr",Temp,Temp,0.1, 12345)
					#L.run(int(round(equilibrationt/L.eval("dt"))))	
					#thermalize(Temp)
					#L.unfix("thermalize")
					
					#print("velocity after equilibration: {}".format(current_atom_handle.velocity))
			else:
				print("displacement not confirmed")
		else:
			print("no displacement found.")
	else:
		print("sputter event detected, equilibration...")
		displacement = True
		sputteringevents[0,nosputt] = nel;		# electron number
		sputteringevents[1,nosputt] = theta;	# scattering angle chosen
		sputteringevents[2,nosputt] = E_t;		# transferred Energy to atom
		sputteringevents[3,nosputt] = atomid;	# atom chosen for energy transfer
		#sputteringevents[4,nosputt] = partdisp;	# atom lost
		#sputteringevents[5,nosputt] = 1;		# element1
		nosputt += 1

		if resetgeom == False:
			L.run(500)
			#L.fix("thermalize","clustergr","temp/csvr",Temp,Temp,0.1, 12345)
			#L.run(int(round(equilibrationt/L.eval("dt"))))	
			#thermalize(Temp)
			#L.unfix("thermalize")
			
			#print("velocity after equilibration: {}".format(current_atom_handle.velocity))

	print("T after equilibration: {} K; dt after equilibration: {} fs".format(L.eval("temp"),L.eval("dt")*1000))
	print("potential energy: {} eV, kinetic energy: {}".format(L.eval("pe"),L.eval("ke")))

	#forces = L.extract_atom("forces",3)      	# lmp.extract_atom(name,type) 
										# extract a per-atom quantity
                                        # name = "x", "type", etc
                                        # type = 0 = vector of ints
                                        #        1 = array of ints
                                        #        2 = vector of doubles
                                        #        3 = array of doubles
	# Close full putput file:
	if fullsavedata > 0:		
		L.undump("savefullstep")	
		print("full dynamics file saved.")
	
	if displacement == True and resetgeom == False:
		#Output final result after each scattering event:
		L.dump(nel,"all","xyz",10000000000, "*" + "T" + str(Temp) + "elno_" + str(nel) + "_EndOut.xyz")
		L.dump_modify(nel,"first","yes","pad",10)	# Schreib beim nullten Schritt ein File raus
		L.dump_modify(nel,"element","Au","Ni","C")
		L.dump_modify(nel,"pbc","yes") # remap atoms via periodic boundary conditions
		L.run(0)
		L.undump(nel)	# Nur ein file pro electron
		(coordsNN11,coordsNN12,coordsNN21,coordsNN22) = calccoordNN(coords1,coords2, NNthresh)
	
	if resetgeom == True:
		resetcoords(startcoords1,startcoords2,indices1,indices2)
		print("geometry reset...")
	
	print("elapsed time: {} s".format(time.time() - starttime))
	print("-------------------------------------------------")
	
	scipy.io.savemat('Output_electronlog.mat', {'shotel':shotelectrons_temp,'sputtering_list':sputteringevents,'displacementthreshold':dispthresh,'high_tension':HT,'element1':"Au",'element2':"Ni",'Startcoordsel1':startcoords1,'Startcoordsel2':startcoords2,'Substratecoords':coordssub,'lammpsindicesel1':startindices1,'lammpsindicesel2':startindices2,'coordnoNN11':coordsNN11,'coordnoNN12':coordsNN12,'coordnoNN21':coordsNN21,'coordnoNN22':coordsNN22})
	
	if math.fmod(nel,100)==0:
		copyfile('Output_electronlog.mat', 'Output_electronlog_backup.mat')
	
L.close()
MPI.Finalize()
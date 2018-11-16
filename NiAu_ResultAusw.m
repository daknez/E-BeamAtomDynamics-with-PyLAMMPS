close all
clear all

standardpath = 'E:\lxshare\';
addpath('P:\projects\projects by number\136xx\13611_Diss Knez_ho\Software\Skripten, Plugins\Eigene\MATLAB\Functions');

[file,filepath] = uigetfile({'*.mat'},'Select Result-file',standardpath);

element1 = 'Au';
element2 = 'Ni';
element3 = 'C';

plotel1 = true;
plotel2 = true;
plotel3 = true;

histogramcategories = 20;
hightension = 300;
calccoordnofromgeometryfile = 'Startgeom.xyz'; % 'Startgeom.xyz';   % set to [] to use coordinationno. from log file

currdens = 6.8600e+26; % j=e-/t * 1/A, 78 pA=4.9e8e-/s (with A... projected area: 1.4 nm2 for 55 atom cluster Ag&Au)

% Constants:
kB = 1.3806488E-23;     % Botlzmann Constant in J/K
elch = 1.602176565E-19; % elementary charge in C
mel = 9.10938291E-31;	% electron mass in kg
u = 1.660538E-27;		% unified atomic mass unit in kg
c = 299792458;          % speed of light m/s
a0 = 0.52917721067e-10;   % Bohr radius in m
barn = 1e-28;           % definition of unit barn in m²
Eel0 = mel*c^2;         

% Material parameters:
masses{1} = {'mAu','mAg','mNi'};
masses{2} = {3.27E-25,107.86*u,58.69*u};
indmass = find(ismember(masses{1},['m' element1]));
mass1 = masses{2}{indmass};
indmass = find(ismember(masses{1},['m' element2]));
mass2 = masses{2}{indmass};

Eel = double(elch) .* double(hightension) .* 1000;
Emax1 = 2*Eel*(Eel+2*Eel0)/(c^2*mass1) / elch;
Emax2 = 2*Eel*(Eel+2*Eel0)/(c^2*mass2) / elch;

thetamax = 180;

%----------------------------------
% read result files
%----------------------------------

load([filepath file]);

if ~isempty(calccoordnofromgeometryfile)

    fileID = fopen([filepath calccoordnofromgeometryfile],'r');
    datacell = textscan(fileID, '%s%f%f%f', 'HeaderLines', 3, 'CollectOutput', 1);
    %datacell = textscan(fileID, '%f%f%f%f%f%f%f%f%f%f%f', 'HeaderLines', 9, 'CollectOutput', 1);
    fclose(fileID);

    coords = datacell{2}';
    species = datacell{1};

    coords1 = coords(:,ismember(species, element1));
    coords2 = coords(:,ismember(species, element2));
    coordssub = coords(:,ismember(species, element3));

    nPart1 = size(coords1,2);
    nPart2 = size(coords2,2);
    nPart = size(coords,2); % nPart1+nPart2;
    nPartsub = size(coordssub,2);
    
end

coords1 = Startcoordsel1';
coords2 = Startcoordsel2';
nPart1 = size(coords1,2);
nPart2 = size(coords2,2);

atomid = shotel(1,:);
E_t = shotel(2,:);
EteV = E_t/elch;    % convert J to eV
dr = shotel(3,:);
theta = shotel(5,:);
theta = theta/pi*180;   % convert rad to deg
atype = shotel(6,:);
NNelement1 = shotel(7,:);
NNelement2 = shotel(8,:);
NNtotal = NNelement1+NNelement2;

displmask = dr>0;
displelement1 = atype==1 & displmask;
displelement2 = atype==2 & displmask;

noscatteringeventselement1 = nnz(atype==1)
noscatteringeventselement2 = nnz(atype==2)

noofdisplelement1 = nnz(displelement1==1)
noofdisplelement2 = nnz(displelement2==1)

Pdisplelement1 = noofdisplelement1/noscatteringeventselement1
Pdisplelement2 = noofdisplelement2/noscatteringeventselement2
Pdispltotal = (noofdisplelement1+noofdisplelement2)/(noscatteringeventselement1+noscatteringeventselement2)

nodispleventsperatomel1 = zeros(nPart1,1);
nodispleventsperatomel2 = zeros(nPart2,1);
noscattereventsperatomel1 = zeros(nPart1,1);
noscattereventsperatomel2 = zeros(nPart2,1);

displacedatomsid = atomid(displmask);
displacedatomtype = atype(displmask);

for stepno=1:length(displacedatomsid)
   if displacedatomtype(stepno) == 1
       ind = find(lammpsindicesel1==displacedatomsid(stepno));
       nodispleventsperatomel1(ind) = nodispleventsperatomel1(ind) + 1;
   elseif displacedatomtype(stepno) == 2
       ind = find(lammpsindicesel2==displacedatomsid(stepno));
       nodispleventsperatomel2(ind) = nodispleventsperatomel2(ind) + 1;
   end
end

for stepno=1:length(atomid)
   if atype(stepno) == 1
       ind = find(lammpsindicesel1==atomid(stepno));
       noscattereventsperatomel1(ind) = noscattereventsperatomel1(ind) + 1;
   elseif atype(stepno) == 2
       ind = find(lammpsindicesel2==atomid(stepno));
       noscattereventsperatomel2(ind) = noscattereventsperatomel2(ind) + 1;
   end
end

Pperatomel1 = nodispleventsperatomel1./noscattereventsperatomel1;
Pperatomel2 = nodispleventsperatomel2./noscattereventsperatomel2;

binvalues = [0:1:12];   % for coordination number histograms

figure;
histogram(NNelement1(displmask),binvalues)
title('Number of nearest Au neighbours for displaced atom')

figure;
histogram(NNelement2(displmask),binvalues)
title('Number of nearest Ni neighbours for displaced atom')

figure;
histogram(NNelement1(displelement1),binvalues)
title('Number of nearest Au neighbours for displaced Au atom')

figure;
histogram(NNelement2(displelement1),binvalues)
title('Number of nearest Ni neighbours for displaced Au atom')

figure;
histogram(NNelement1(displelement2),binvalues)
title('Number of nearest Au neighbours for displaced Ni atom')

figure;
histogram(NNelement2(displelement2),binvalues)
title('Number of nearest Ni neighbours for displaced Ni atom')

figure;
histogram(NNtotal(displmask),binvalues)
title('Number of nearest neighbours for displaced atom')

figure;
histogram(NNtotal(NNtotal~=0),binvalues)
title('Number of nearest neighbours for all atoms')

% create theta histogram:
thetahist= 0:thetamax/histogramcategories:thetamax;
thetahist = [thetahist inf];

displacementshist = histc(theta(displmask),thetahist);
totalhist = histc(theta,thetahist);
thetahistval = displacementshist./totalhist;
thetahistval(isnan(thetahistval)) = 0; % delete NaN
thetahistval = thetahistval(1:end-2);
thetahistcentered = thetahist+(thetamax/histogramcategories)/2;
thetahistcentered = thetahistcentered(1:end-2);

thetahistval1 = histc(theta(displelement1),thetahist)./totalhist;
thetahistval1(isnan(thetahistval1)) = 0; % delete NaN
thetahistval1 = thetahistval1(1:end-2);
thetahistval2 = histc(theta(displelement2),thetahist)./totalhist;
thetahistval2(isnan(thetahistval2)) = 0; % delete NaN
thetahistval2 = thetahistval2(1:end-2);

figdispratetheta = figure;
plot(thetahistcentered,thetahistval,'*')

figdispratethetaperelement = figure;
plot(thetahistcentered,thetahistval1,'o')
hold on;
plot(thetahistcentered,thetahistval2,'x')
hold off;
legend('Au','Ni','Location','northwest')

figtheta = figure;
b3 = bar(thetahist(1:end-1),histc(theta,thetahist(1:end-1)));
set(b3,'FaceColor','white');
hold on;
eb3 = errorbar(thetahist(1:end-1),histc(theta,thetahist(1:end-1)),sqrt(histc(theta,thetahist(1:end-1))),'.');
set(eb3,'color','blue');

b4 = bar(thetahist(1:end-1),histc(theta(displmask),thetahist(1:end-1)),'r');
set(b4,'FaceColor','red');
eb4 = errorbar(thetahist(1:end-1),histc(theta(displmask),thetahist(1:end-1)),sqrt(histc(theta(displmask),thetahist(1:end-1))),'.');
set(eb4,'color','blue');
hold off;
ylim([0 2*max(histc(theta(displmask),thetahist))]);
title('Histogram of number of displaced atoms per scattering angle \theta','interpreter','tex');
xlabel('\theta [°]');
ylabel('number of displaced atoms');
hold off;

% create E_t histogram:
Ethist = 0:Emax1/histogramcategories:Emax1;
Ethist = [Ethist inf];
Ehistval = histc(EteV(displmask),Ethist)./histc(EteV,Ethist);
Ethistcentered1 = Ethist+(Emax1/histogramcategories)/2;
Ethistcentered1 = Ethistcentered1(1:end-2);

figEthist = figure;
b1 = bar(Ethist(1:end-1),histc(EteV,Ethist(1:end-1)));
set(b1,'FaceColor','white','EdgeColor','black');

hold on;
b2 = bar(Ethist(1:end-1),histc(EteV(displmask),Ethist(1:end-1)),'r');
set(b2,'FaceColor','red');
hold off;
ylim([0 2*max(histc(EteV(displmask),Ethist(1:end-1)))]);
title('Histogram of transferred energy E_t per displaced atom','interpreter','tex');
xlabel('E_t [eV]');
ylabel('number of displaced atoms');
hold off;

% Calculate scattering cross section from NIST tables:
filestring = strcat(filepath,[element1 num2str(hightension)]);
scatter_temp = load(filestring);
crosssections1 = [scatter_temp.deg scatter_temp.cross_section];
filestring = strcat(filepath,[element2 num2str(hightension)]);
scatter_temp = load(filestring);
crosssections2 = [scatter_temp.deg scatter_temp.cross_section];

%reduce original NIST cross section to histogram sampling:
for i = 1:length(thetahistcentered)
    [~,inda] = min(abs(crosssections1(:,1)-thetahistcentered(i)));
    reducedcross1(i) = crosssections1(inda,2);
    [~,inda] = min(abs(crosssections2(:,1)-thetahistcentered(i)));
    reducedcross2(i) = crosssections2(inda,2);
end

totalcross1 = 2*pi*trapz(crosssections1(:,1)./180*pi,crosssections1(:,2).*sin(crosssections1(:,1)./180*pi)).*a0^2
totalcross2 = 2*pi*trapz(crosssections2(:,1)./180*pi,crosssections2(:,2).*sin(crosssections2(:,1)./180*pi)).*a0^2

% figure;
% plot(thetahistcentered,reducedcross1,'o',crosssections1(:,1),crosssections1(:,2));
% hold on
% plot(thetahistcentered,reducedcross2,'x',crosssections2(:,1),crosssections2(:,2));
% hold off

figdiplcross = figure;
semilogy(thetahistcentered, reducedcross1.*thetahistval1,'x')
hold on
semilogy(thetahistcentered, reducedcross2.*thetahistval2,'*')
hold off
legend('Au','Ni','Location','northwest')

totaldispcrosssec = 2*pi*trapz(thetahistcentered./180*pi,thetahistval.*reducedcross1.*sin(thetahistcentered./180*pi)).*a0^2
t = 1/(currdens*totaldispcrosssec)

dispcrosssec1 = 2*pi*trapz(thetahistcentered./180*pi,thetahistval1.*reducedcross1.*sin(thetahistcentered./180*pi)).*a0^2
dispcrosssec2 = 2*pi*trapz(thetahistcentered./180*pi,thetahistval2.*reducedcross1.*sin(thetahistcentered./180*pi)).*a0^2

t1 = 1/(currdens*dispcrosssec1)
t2 = 1/(currdens*dispcrosssec2)

% Plot 3D:
az_el = [45 25];
rel1 = 1.44;
rel2 = 1.24;
rel3 = 0.7;
borderw = 1;
rb = 1;

min_x = min(coords(1,:));
max_x = max(coords(1,:));
max_y = max(coords(2,:));
min_y = min(coords(2,:));
max_z = max(coords(3,:))+10;
min_z = min(coords(3,:));

min_E = 0;
max_E = 0.5;

% find centre atom:
centre = [max_x-min_x max_y-min_y max_z-min_z]; % find centre
drc = coords(:,:,1)-repmat(centre',1,nPart);    % calculate distances of atoms ot centre
drc = sqrt(drc(1,:).^2 + drc(2,:).^2+drc(3,:).^2);
centreatomnr = find(drc == min(drc));
centreatomx = coords(1,centreatomnr,1);
centreatomy = coords(2,centreatomnr,1);

% min_x = min(coords(1,:,1))-centerofmass(1,1);
% max_x = max(coords(1,:,1))-centerofmass(1,1);
% max_y = max(coords(2,:,1))-centerofmass(2,1);
% min_y = min(coords(2,:,1))-centerofmass(2,1);

% min_x = -1.5E-9;
% max_x = +1.5E-9;
% max_y = +1.5E-9;
% min_y = -1.5E-9;
% max_z = 2E-9;
% min_z = subz;


plotpartpos = figure('Visible','On');
%plotpartpos = myaa;

set(gcf, 'renderer', 'opengl');
[x,y,z] = sphere(16);

[Xgauss,Ygauss]=meshgrid(linspace(min_x-borderw,max_x+borderw,256),linspace(min_y-borderw,max_y+borderw,256));  % //mesh
gaussz   = zeros(size(Xgauss));

%str = {['Time: ',num2str(round(time(step)*1e12)),' ps'],['Temperature: ',num2str(sprintf('%.1f',T(step))),' K']};
%str ='TEST'
%title(strcat('time step no: ',num2str(step*printFreq),'; real time: ',num2str(time(step)),'s, Temperature: ',num2str(T(step)),' K'));
%annotation('textbox', [0, 0.1, 0.4, 0], 'string',str,'FontSize',14)

%% plot substrate atoms:
ax2 = axes;
if plotel3
    for ipt=1:nPartsub
            atomx = coordssub(1,ipt);%-centreatomx;
            atomy = coordssub(2,ipt);%-centreatomy;

            atomz = coordssub(3,ipt);
            %c = ones(size(z))*EkineV(ipt);
            %c = ones(size(z))*countNN(ipt);
            c = ones(size(z));

            % draw spheres:
            s = surface(ax2,rel3*x+atomx,rel3*y+atomy,rel3*z+atomz,c);
            set(s,'EdgeColor','none');

            % draw gaussian projections:
            gaussz= gaussz + 0.05 * exp(-((Xgauss-atomx).^2+(Ygauss-atomy).^2)/( 2.*rb.^2) );

            %hcoord = ownhistogram(hcoord,countNN(ipt,step)); % histogram for coordination numbers
    end
end

ax1 = axes;
if plotel1
    for ipt=1:nPart1 % run over all atoms in one timestep
        %if coords1(1,ipt,step) ~= 0

            atomx = coords1(1,ipt);%-centreatomx;
            atomy = coords1(2,ipt);%-centreatomy;

            atomz = coords1(3,ipt);
            %c = ones(size(z))*EkineV(ipt);
            %c = ones(size(z))*countNN(ipt,step);
            %c = ones(size(z))*nodispleventsperatomel1(ipt);
            %c = ones(size(z))*noscattereventsperatomel1(ipt);
            %c = ones(size(z))*Pperatomel1(ipt);
            c = ones(size(z));

            % draw spheres:
            s = surface(ax1,rel1*x+atomx,rel1*y+atomy,rel1*z+atomz,c);
            set(s,'EdgeColor','none');

            % draw gaussian projections:
            gaussz= gaussz + exp(-((Xgauss-atomx).^2+(Ygauss-atomy).^2)/( 2.*rb.^2) );

            %hcoord = ownhistogram(hcoord,countNN(ipt,step)); % histogram for coordination numbers
    end
end

if plotel2
    for ipt=1:nPart2
            atomx = coords2(1,ipt);%-centreatomx;
            atomy = coords2(2,ipt);%-centreatomy;

            atomz = coords2(3,ipt);
            %c = ones(size(z))*EkineV(ipt);
            %c = ones(size(z))*countNN(ipt);
            %c = ones(size(z))*nodispleventsperatomel2(ipt);
            %c = ones(size(z))*noscattereventsperatomel2(ipt);
            %c = ones(size(z))*Pperatomel2(ipt);
            c = 2.*ones(size(z));

            % draw spheres:
            s = surface(ax1,rel2*x+atomx,rel2*y+atomy,rel2*z+atomz,c);
            set(s,'EdgeColor','none');

            % draw gaussian projections:
            gaussz= gaussz + 0.2 * exp(-((Xgauss-atomx).^2+(Ygauss-atomy).^2)/( 2.*rb.^2) );

            %hcoord = ownhistogram(hcoord,countNN(ipt,step)); % histogram for coordination numbers
    end
end

%% Link axis together
linkaxes([ax1,ax2])
%% Hide the top axes
ax2.Visible = 'off';
ax2.XTick = [];
ax2.YTick = [];

%% Give each one its own colormap
%colormap(ax1,'jet')
colormap(ax1,[[255 215 0]./255;[200 200 200]./255])  % Gold and Silver
colormap(ax2,[50 0 0]./255)% Brown
axis([ax1,ax2],'equal')

% bring axis on top again (fix matlab bug)
%set([ax1,ax2],'Layer', 'top');
set(ax1, 'Color', 'none')

%% Then add colorbars and get everything lined up
set([ax1,ax2],'Position',[.17 .11 .685 .815]);
%cb1 = colorbar(ax2,'Position',[.05 .11 .0675 .815]);
cb2 = colorbar(ax1,'Position',[.88 .11 .0675 .815]);

%cb = colorbar(ax1); 
%colormap('jet');
%colormap([200 200 200 ; 50 0 0]./255)
%colorm = [200 200 200]./255;
%colorm = [jet(256);hot(256)];
%colorm = [[200 200 200]./255;jet(256)];
%colormap(gca,colorm)
xlim([ax1,ax2],[min_x-borderw max_x+borderw]) 
ylim([ax1,ax2],[min_y-borderw max_y+borderw]) 
zlim([ax1,ax2],[min_z-borderw max_z+borderw]) 
%caxis([min_E max_E])
%caxis(ax1,[0 max(nodispleventsperatomel2)])
xlabel(ax1,'x  [m]');
ylabel(ax1,'y  [m]');
zlabel(ax1,'z  [m]');
%ylabel(cb,'kinetic Energy [eV]')
%ylabel(cb,'Number of nearest neighbours')
hold on;
view(ax1,az_el);
view(ax2,az_el);
light(ax1)
light(ax2)
lighting(ax1,'gouraud')
lighting(ax2,'gouraud')

hold off;


%% draw projected HAADF intensity-image
planeimg = abs(gaussz);
% scale image between [0, 255] in order to use a custom color map for it.
minplaneimg = min(min(planeimg)); % find the minimum
scaledimg = (floor(((planeimg - minplaneimg)./(max(max(planeimg)) - minplaneimg)) * 255)); % perform scaling
% convert the image to a true color image with colormap.
colorimg = ind2rgb(scaledimg,gray(256));

figADF = figure('Visible','On');
ADFimg = imagesc([min_x-borderw max_x+borderw],[min_y-borderw max_y+borderw],colorimg);
axis equal
title(['\bf{"Simulated" HAADF-image scaled to intensity 1}'],'interpreter','tex');
xlabel('x  [nm]');
ylabel('y  [nm]');
%colormap('gray');
%ADFcb = colorbar; 

% In Tabellenform bringen:
thetahistval1 = thetahistval1';
thetahistval2 = thetahistval2';
thetahistcentered = thetahistcentered';
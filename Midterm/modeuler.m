function [fo]=modeuler(h,old,a,b,c)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Purpose: Apply modified Euler scheme to the Lorenz equations
%
%  (c) 2002  Data Assimilation Research Centre
%
%  Original Fortran program by Marek Wlasak
%  Converted to Matlab by Amos Lawless
%
%  6/7/04 Moved all functions inline to make adjoint easier
%
%  List of main variables
%    a:          A coefficient in equations
%    b:          B coefficient in equations
%    c:          C coefficient in equations
%    h:          Time step for numerical scheme
%    tstep:      Number of time steps to perform
%    [xval,yval,zval]: Initial fields
%    j:          Index to pick up correct initial field
%
%  Output:
%    [xo,yo,zo]: Trajectories of evolved fields
%
% modified by T. Matsuo for the course 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
x(1)=old(1);
y(1)=old(2);
z(1)=old(3);

kx1 = h*a*(y-x);
ky1 = h*(b*x-y-x*z);
kz1 = h*(x*y- c*z);

kx2 = h*a*(y+ky1-x-kx1);
ky2 = h*(b*(x+kx1)-y-ky1-(x+kx1)*(z+kz1));
kz2 = h*((x+kx1)*(y+ky1)-c*(z+kz1));

x=x+0.5d0*(kx1+kx2);
y=y+0.5d0*(ky1+ky2);
z=z+0.5d0*(kz1+kz2);

fo(:,1)=x;
fo(:,2)=y;
fo(:,3)=z;
%


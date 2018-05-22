function [im, dSigmaT, dAlbedo, dGVal] = renderDerivImageWeight(sigmaT, albedo, gVal,...
						samplingSigmaT, samplingAlbedo, samplingGVal,...	
						scene, renderer)
%% 
% All units are in mm.

% sample
iorMedium = scene.iorMedium;
mediumDimensions = scene.mediumDimensions;

% light source
lightOrigin = scene.lightOrigin;
lightDir = scene.lightDir;
lightPlane = scene.lightPlane;
Li = scene.Li;

% camera and image
viewOrigin = scene.viewOrigin;
viewDir = scene.viewDir;
viewX = scene.viewX;
viewY = scene.viewY;
viewPlane = scene.viewPlane;
pathlengthRange = scene.pathlengthRange;
viewReso = scene.viewReso;

% renderer
numPhotons = renderer.numPhotons;
maxDepth = renderer.maxDepth;
maxPathlength = renderer.maxPathlength;
useDirect = renderer.useDirect;

%%
[im, dSigmaT, dAlbedo, dGVal] = renderDerivImageWeight_mex(sigmaT, albedo, gVal, ...
					samplingSigmaT, samplingAlbedo, samplingGVal,...
					iorMedium, mediumDimensions, ...
					lightOrigin, lightDir, lightPlane, Li, ...
					viewOrigin, viewDir, viewX, viewY, viewPlane, pathlengthRange, viewReso, ...
					numPhotons, maxDepth, maxPathlength, useDirect);
im = permute(im, [2 1 3]);
dSigmaT = permute(dSigmaT, [2 1 3]);
dAlbedo = permute(dAlbedo, [2 1 3]);
dGVal = permute(dGVal, [2 1 3]);

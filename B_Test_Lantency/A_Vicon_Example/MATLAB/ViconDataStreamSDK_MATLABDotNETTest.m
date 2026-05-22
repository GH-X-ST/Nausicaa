% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright (C) OMG Plc 2009.
% All rights reserved.  This software is protected by copyright
% law and international treaties.  No part of this software / document
% may be reproduced or distributed in any form or by any means,
% whether transiently or incidentally to some other use of this software,
% without the written permission of the copyright owner.
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part of the Vicon DataStream SDK for MATLAB.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Program options
TransmitMulticast = false;
EnableHapticFeedbackTest = false;
HapticOnList = {'ViconAP_001';'ViconAP_002'};
SubjectFilterApplied = false;
bPrintSkippedFrame = false;

% Check whether these variables exist, as they can be set by the command line on launch
% If you run the script with Command Window in Matlab, these workspace vars could persist the value from previous runs even not set in the Command Window
% You could clear the value with "clearvars"
if ~exist( 'bReadCentroids' )
  bReadCentroids = false;
end

if ~exist( 'bReadRays' )
  bReadRays = false;
end

if ~exist( 'bTrajectoryIDs' )
  bTrajectoryIDs = false;
end

if ~exist( 'axisMapping' )
  axisMapping = 'ZUp';
end

% example for running from commandline in the ComandWindow in Matlab
% e.g. bLightweightSegment = true;HostName = 'localhost:801';ViconDataStreamSDK_MATLABTest
if ~exist('bLightweightSegment')
  bLightweightSegment = false;
end

% Pass the subjects to be filtered in
% e.g. Subject = {'Subject1'};HostName = 'localhost:801';ViconDataStreamSDK_MATLABTest
EnableSubjectFilter  = exist('subjects');

% Program options
if ~exist( 'HostName' )
  HostName = 'localhost:801';
end

if exist('undefVar')
  fprintf('Undefined Variable: %s\n', mat2str( undefVar ) );
end

fprintf( 'Centroids Enabled: %s\n', mat2str( bReadCentroids ) );
fprintf( 'Rays Enabled: %s\n', mat2str( bReadRays ) );
fprintf( 'Trajectory IDs Enabled: %s\n', mat2str( bTrajectoryIDs ) );
fprintf( 'Lightweight Segment Data Enabled: %s\n', mat2str( bLightweightSegment ) );
fprintf('Axis Mapping: %s\n', axisMapping )

% Load the SDK
fprintf( 'Loading SDK...' );
addpath( '..\dotNET' );
dssdkAssembly = which('ViconDataStreamSDK_DotNET.dll');
if dssdkAssembly == ""
  [ file, path ] = uigetfile( '*.dll' );
  if isequal( file, 0 )
    fprintf( 'User canceled' );
    return;
  else
    dssdkAssembly = fullfile( path, file );
  end   
end

NET.addAssembly(dssdkAssembly);
fprintf( 'done\n' );

% A dialog to stop the loop
MessageBox = msgbox( 'Stop DataStream Client', 'Vicon DataStream SDK' );

% Make a new client
MyClient = ViconDataStreamSDK.DotNET.Client();

% Connect to a server
fprintf( 'Connecting to %s ...', HostName );
while ~MyClient.IsConnected().Connected
  % Direct connection
  MyClient.Connect( HostName );
  
  % Multicast connection
  % MyClient.ConnectToMulticast( HostName, '224.0.0.0' );
  
  fprintf( '.' );
end
fprintf( '\n' );

% Enable some different data types
MyClient.EnableSegmentData();
MyClient.EnableMarkerData();
MyClient.EnableUnlabeledMarkerData();
MyClient.EnableDeviceData();
if bReadCentroids
  MyClient.EnableCentroidData();
end
if bReadRays
  MyClient.EnableMarkerRayData();
end

if bLightweightSegment
  MyClient.DisableLightweightSegmentData();
  Output_EnableLightweightSegment = MyClient.EnableLightweightSegmentData();
  if Output_EnableLightweightSegment.Result ~= ViconDataStreamSDK.DotNET.Result.Success
    fprintf( 'Server does not support lightweight segment data.\n' );
  end
end

fprintf( 'Segment Data Enabled: %s\n',          AdaptBool( MyClient.IsSegmentDataEnabled().Enabled ) );
fprintf( 'Marker Data Enabled: %s\n',           AdaptBool( MyClient.IsMarkerDataEnabled().Enabled ) );
fprintf( 'Unlabeled Marker Data Enabled: %s\n', AdaptBool( MyClient.IsUnlabeledMarkerDataEnabled().Enabled ) );
fprintf( 'Device Data Enabled: %s\n',           AdaptBool( MyClient.IsDeviceDataEnabled().Enabled ) );
fprintf( 'Centroid Data Enabled: %s\n',         AdaptBool( MyClient.IsCentroidDataEnabled().Enabled ) );
fprintf( 'Marker Ray Data Enabled: %s\n',       AdaptBool( MyClient.IsMarkerRayDataEnabled().Enabled ) );

MyClient.SetBufferSize(1)
% % Set the streaming mode
MyClient.SetStreamMode( ViconDataStreamSDK.DotNET.StreamMode.ClientPull  );
% % MyClient.SetStreamMode( StreamMode.ClientPullPreFetch );
% % MyClient.SetStreamMode( StreamMode.ServerPush );

% % Set the global up axis
if axisMapping == 'XUp'
  MyClient.SetAxisMapping( ViconDataStreamSDK.DotNET.Direction.Up, ...
                           ViconDataStreamSDK.DotNET.Direction.Forward,      ...
                           ViconDataStreamSDK.DotNET.Direction.Left ); % X-up
elseif axisMapping == 'YUp'
  MyClient.SetAxisMapping(  ViconDataStreamSDK.DotNET.Direction.Forward, ...
                          ViconDataStreamSDK.DotNET.Direction.Up,    ...
                          ViconDataStreamSDK.DotNET.Direction.Right );    % Y-up
else
  MyClient.SetAxisMapping(  ViconDataStreamSDK.DotNET.Direction.Forward, ...
                          ViconDataStreamSDK.DotNET.Direction.Left,    ...
                          ViconDataStreamSDK.DotNET.Direction.Up );    % Z-up
end

Output_GetAxisMapping = MyClient.GetAxisMapping();
fprintf( 'Axis Mapping: X-%s Y-%s Z-%s\n', char( Output_GetAxisMapping.XAxis.ToString() ), ...
                                           char( Output_GetAxisMapping.YAxis.ToString() ), ...
                                           char( Output_GetAxisMapping.ZAxis.ToString() ) );

% Discover the version number
Output_GetVersion = MyClient.GetVersion();
fprintf( 'Version: %d.%d.%d\n', Output_GetVersion.Major, ...
                                Output_GetVersion.Minor, ...
                                Output_GetVersion.Point );
  
if TransmitMulticast
  MyClient.StartTransmittingMulticast( 'localhost', '224.0.0.0' );
end  

Frame = -1;
SkippedFrames = [];
Counter = 1;

tStart = tic;
% Loop until the message box is dismissed
while ishandle( MessageBox )
  drawnow limitrate;
  Counter = Counter + 1;
  
  % Get a frame
  fprintf( 'Waiting for new frame...' );
  while MyClient.GetFrame().Result ~= ViconDataStreamSDK.DotNET.Result.Success
    fprintf( '.' );
  end% while
  fprintf( '\n' );  
  
  if EnableHapticFeedbackTest
    if mod( Counter,2 ) == 0
      for i = 0:length( HapticOnList ) -1
          DeviceName = HapticOnList{i};
          Output_GetApexFeedback = MyClient.SetApexDeviceFeedback( DeviceName, true );
          if Output_GetApexFeedback.Result == ViconDataStreamSDK.DotNET.Result.Success
              fprintf( 'Turn haptic feedback on for device: %s\n', DeviceName );
          elseif Output_GetApexFeedback.Result == Result.InvalidDeviceName
              fprintf( 'Device doesn''t exist: %s\n', DeviceName );
          end
      end
    end
    if mod( Counter, 20 ) == 0
      for i = 0:length( HapticOnList ) -1
          DeviceName = HapticOnList{i};
          Output_GetApexFeedback = MyClient.SetApexDeviceFeedback( DeviceName, false );
          if Output_GetApexFeedback.Result == ViconDataStreamSDK.DotNET.Result.Success
              fprintf( 'Turn haptic feedback on for device: %s\n', DeviceName );
          end
      end
    end
  end
      

  % Get the frame number
  Output_GetFrameNumber = MyClient.GetFrameNumber();
  if Frame ~= -1
    while Output_GetFrameNumber.FrameNumber > Frame + 1
      SkippedFrames = [SkippedFrames Frame+1];
      if bPrintSkippedFrame
        fprintf( 'Skipped frame: %d\n', Frame+1 );      
      end
      Frame = Frame + 1;
    end
  end
  Frame = Output_GetFrameNumber.FrameNumber;  
  fprintf( 'Frame Number: %d\n', Output_GetFrameNumber.FrameNumber );

  % Get the frame rate
  Output_GetFrameRate = MyClient.GetFrameRate();
  fprintf( 'Frame rate: %g\n', Output_GetFrameRate.FrameRateHz );

  for FrameRateIndex = 0:MyClient.GetFrameRateCount().Count -1
    FrameRateName  = MyClient.GetFrameRateName( FrameRateIndex ).Name;
    FrameRateValue = MyClient.GetFrameRateValue( FrameRateName ).Value;

    fprintf( '%s: %gHz\n', char(FrameRateName), FrameRateValue );
  end% for  

  fprintf( '\n' );
  % Get the timecode
  Output_GetTimecode = MyClient.GetTimecode();
  fprintf( 'Timecode: %dh %dm %ds %df %dsf %s %d %d %d\n\n',    ...
                     Output_GetTimecode.Hours,                  ...
                     Output_GetTimecode.Minutes,                ...
                     Output_GetTimecode.Seconds,                ...
                     Output_GetTimecode.Frames,                 ...
                     Output_GetTimecode.SubFrame,               ...
                     AdaptBool( Output_GetTimecode.FieldFlag ), ...
                     double(Output_GetTimecode.Standard),               ...
                     Output_GetTimecode.SubFramesPerFrame,      ...
                     Output_GetTimecode.UserBits );

  % Get the latency
  fprintf( 'Latency: %gs\n', MyClient.GetLatencyTotal().Total );
  
  for LatencySampleIndex = 0:typecast( MyClient.GetLatencySampleCount().Count, 'int32' ) -1
    SampleName  = MyClient.GetLatencySampleName( typecast( LatencySampleIndex, 'uint32') ).Name;
    SampleValue = MyClient.GetLatencySampleValue( SampleName ).Value;

    fprintf( '  %s %gs\n', char(SampleName), SampleValue );
  end% for  
  fprintf( '\n' );
                     
  Output_GetHardwareFrameNumber = MyClient.GetHardwareFrameNumber();
  fprintf( 'Hardware Frame Number: %d\n', Output_GetHardwareFrameNumber.HardwareFrameNumber );

  if EnableSubjectFilter && ~SubjectFilterApplied 
    for SubjectIndex = 1: length( Subject )
      Output_SubjectFilter = MyClient.AddToSubjectFilter(char( Subject(SubjectIndex)));
      SubjectFilterApplied = SubjectFilterApplied || Output_SubjectFilter.Result.Value == Result.Success;
    end
  end

  % Count the number of subjects
  SubjectCount = MyClient.GetSubjectCount().SubjectCount;
  fprintf( 'Subjects (%d):\n', SubjectCount );
  for SubjectIndex = 0:typecast( SubjectCount, 'int32' ) -1
    fprintf( '  Subject #%d\n', SubjectIndex );
    
    % Get the subject name
    SubjectName = MyClient.GetSubjectName( typecast( SubjectIndex, 'uint32') ).SubjectName;
    fprintf( '    Name: %s\n', char(SubjectName) );
    
    % Get the root segment
    RootSegment = MyClient.GetSubjectRootSegmentName( SubjectName ).SegmentName;
    fprintf( '    Root Segment: %s\n', char(RootSegment) );

    % Count the number of segments
    SegmentCount = MyClient.GetSegmentCount( SubjectName ).SegmentCount;
    fprintf( '    Segments (%d):\n', SegmentCount );
    for SegmentIndex = 0:typecast( SegmentCount , 'int32' )-1
      fprintf( '      Segment #%d\n', SegmentIndex );
      
      % Get the segment name
      SegmentName = MyClient.GetSegmentName( SubjectName, typecast( SegmentIndex, 'uint32') ).SegmentName;
      fprintf( '        Name: %s\n', char( SegmentName ) );
      
      % Get the segment parent
      SegmentParentName = MyClient.GetSegmentParentName( SubjectName, SegmentName ).SegmentName;
      fprintf( '        Parent: %s\n',  char( SegmentParentName ) );

      % Get the segment's children
      ChildCount = MyClient.GetSegmentChildCount( SubjectName, SegmentName ).SegmentCount;
      fprintf( '     Children (%d):\n', ChildCount );
      for ChildIndex = 0:typecast( ChildCount, 'int32' )-1
        ChildName = MyClient.GetSegmentChildName( SubjectName, SegmentName, typecast( ChildIndex, 'uint32' ) ).SegmentName;
        fprintf( '       %s\n', char( ChildName ) );
      end% for  

      % Get the static segment translation
      Output_GetSegmentStaticTranslation = MyClient.GetSegmentStaticTranslation( SubjectName, SegmentName );
      fprintf( '        Static Translation: (%g, %g, %g)\n',                  ...
                         Output_GetSegmentStaticTranslation.Translation( 1 ), ...
                         Output_GetSegmentStaticTranslation.Translation( 2 ), ...
                         Output_GetSegmentStaticTranslation.Translation( 3 ) );
      
      % Get the static segment rotation in helical co-ordinates
      Output_GetSegmentStaticRotationHelical = MyClient.GetSegmentStaticRotationHelical( SubjectName, SegmentName );
      fprintf( '        Static Rotation Helical: (%g, %g, %g)\n',              ...
                         Output_GetSegmentStaticRotationHelical.Rotation( 1 ), ...
                         Output_GetSegmentStaticRotationHelical.Rotation( 2 ), ...
                         Output_GetSegmentStaticRotationHelical.Rotation( 3 ) );
      
      % Get the static segment rotation as a matrix
      Output_GetSegmentStaticRotationMatrix = MyClient.GetSegmentStaticRotationMatrix( SubjectName, SegmentName );
      fprintf( '        Static Rotation Matrix: (%g, %g, %g, %g, %g, %g, %g, %g, %g)\n', ...
                         Output_GetSegmentStaticRotationMatrix.Rotation( 1 ),            ...
                         Output_GetSegmentStaticRotationMatrix.Rotation( 2 ),            ...
                         Output_GetSegmentStaticRotationMatrix.Rotation( 3 ),            ...
                         Output_GetSegmentStaticRotationMatrix.Rotation( 4 ),            ...
                         Output_GetSegmentStaticRotationMatrix.Rotation( 5 ),            ...
                         Output_GetSegmentStaticRotationMatrix.Rotation( 6 ),            ...
                         Output_GetSegmentStaticRotationMatrix.Rotation( 7 ),            ...
                         Output_GetSegmentStaticRotationMatrix.Rotation( 8 ),            ...
                         Output_GetSegmentStaticRotationMatrix.Rotation( 9 ) );
      
      % Get the static segment rotation in quaternion co-ordinates
      Output_GetSegmentStaticRotationQuaternion = MyClient.GetSegmentStaticRotationQuaternion( SubjectName, SegmentName );
      fprintf( '        Static Rotation Quaternion: (%g, %g, %g, %g)\n',          ...
                         Output_GetSegmentStaticRotationQuaternion.Rotation( 1 ), ...
                         Output_GetSegmentStaticRotationQuaternion.Rotation( 2 ), ...
                         Output_GetSegmentStaticRotationQuaternion.Rotation( 3 ), ...
                         Output_GetSegmentStaticRotationQuaternion.Rotation( 4 ) );

      % Get the static segment rotation in EulerXYZ co-ordinates
      Output_GetSegmentStaticRotationEulerXYZ = MyClient.GetSegmentStaticRotationEulerXYZ( SubjectName, SegmentName );
      fprintf( '        Static Rotation EulerXYZ: (%g, %g, %g)\n',               ...
                         Output_GetSegmentStaticRotationEulerXYZ.Rotation( 1 ),  ...
                         Output_GetSegmentStaticRotationEulerXYZ.Rotation( 2 ),  ...
                         Output_GetSegmentStaticRotationEulerXYZ.Rotation( 3 ) );

      % Get the global segment translation
      Output_GetSegmentGlobalTranslation = MyClient.GetSegmentGlobalTranslation( SubjectName, SegmentName );
      fprintf( '        Global Translation: (%g, %g, %g) %s\n',               ...
                         Output_GetSegmentGlobalTranslation.Translation( 1 ), ...
                         Output_GetSegmentGlobalTranslation.Translation( 2 ), ...
                         Output_GetSegmentGlobalTranslation.Translation( 3 ), ...
                         AdaptBool( Output_GetSegmentGlobalTranslation.Occluded ) );
      
      % Get the global segment rotation in helical co-ordinates
      Output_GetSegmentGlobalRotationHelical = MyClient.GetSegmentGlobalRotationHelical( SubjectName, SegmentName );
      fprintf( '        Global Rotation Helical: (%g, %g, %g) %s\n',           ...
                         Output_GetSegmentGlobalRotationHelical.Rotation( 1 ), ...
                         Output_GetSegmentGlobalRotationHelical.Rotation( 2 ), ...
                         Output_GetSegmentGlobalRotationHelical.Rotation( 3 ), ...
                         AdaptBool( Output_GetSegmentGlobalRotationHelical.Occluded ) );
      
      % Get the global segment rotation as a matrix
      Output_GetSegmentGlobalRotationMatrix = MyClient.GetSegmentGlobalRotationMatrix( SubjectName, SegmentName );
      fprintf( '        Global Rotation Matrix: (%g, %g, %g, %g, %g, %g, %g, %g, %g) %s\n', ...
                         Output_GetSegmentGlobalRotationMatrix.Rotation( 1 ),               ...
                         Output_GetSegmentGlobalRotationMatrix.Rotation( 2 ),               ...
                         Output_GetSegmentGlobalRotationMatrix.Rotation( 3 ),               ...
                         Output_GetSegmentGlobalRotationMatrix.Rotation( 4 ),               ...
                         Output_GetSegmentGlobalRotationMatrix.Rotation( 5 ),               ...
                         Output_GetSegmentGlobalRotationMatrix.Rotation( 6 ),               ...
                         Output_GetSegmentGlobalRotationMatrix.Rotation( 7 ),               ...
                         Output_GetSegmentGlobalRotationMatrix.Rotation( 8 ),               ...
                         Output_GetSegmentGlobalRotationMatrix.Rotation( 9 ),               ...
                         AdaptBool( Output_GetSegmentGlobalRotationMatrix.Occluded ) );
      
      % Get the global segment rotation in quaternion co-ordinates
      Output_GetSegmentGlobalRotationQuaternion = MyClient.GetSegmentGlobalRotationQuaternion( SubjectName, SegmentName );
      fprintf( '        Global Rotation Quaternion: (%g, %g, %g, %g) %s\n',             ...
                         Output_GetSegmentGlobalRotationQuaternion.Rotation( 1 ),       ...
                         Output_GetSegmentGlobalRotationQuaternion.Rotation( 2 ),       ...
                         Output_GetSegmentGlobalRotationQuaternion.Rotation( 3 ),       ...
                         Output_GetSegmentGlobalRotationQuaternion.Rotation( 4 ),       ...
                         AdaptBool( Output_GetSegmentGlobalRotationQuaternion.Occluded ) );

      % Get the global segment rotation in EulerXYZ co-ordinates
      Output_GetSegmentGlobalRotationEulerXYZ = MyClient.GetSegmentGlobalRotationEulerXYZ( SubjectName, SegmentName );
      fprintf( '        Global Rotation EulerXYZ: (%g, %g, %g) %s\n',                 ...
                         Output_GetSegmentGlobalRotationEulerXYZ.Rotation( 1 ),       ...
                         Output_GetSegmentGlobalRotationEulerXYZ.Rotation( 2 ),       ...
                         Output_GetSegmentGlobalRotationEulerXYZ.Rotation( 3 ),       ...
                         AdaptBool( Output_GetSegmentGlobalRotationEulerXYZ.Occluded ) );

      % Get the local segment translation
      Output_GetSegmentLocalTranslation = MyClient.GetSegmentLocalTranslation( SubjectName, SegmentName );
      fprintf( '        Local Translation: (%g, %g, %g) %s\n',               ...
                         Output_GetSegmentLocalTranslation.Translation( 1 ), ...
                         Output_GetSegmentLocalTranslation.Translation( 2 ), ...
                         Output_GetSegmentLocalTranslation.Translation( 3 ), ...
                         AdaptBool( Output_GetSegmentLocalTranslation.Occluded ) );
      
      % Get the local segment rotation in helical co-ordinates
      Output_GetSegmentLocalRotationHelical = MyClient.GetSegmentLocalRotationHelical( SubjectName, SegmentName );
      fprintf( '        Local Rotation Helical: (%g, %g, %g) %s\n',           ...
                         Output_GetSegmentLocalRotationHelical.Rotation( 1 ), ...
                         Output_GetSegmentLocalRotationHelical.Rotation( 2 ), ...
                         Output_GetSegmentLocalRotationHelical.Rotation( 3 ), ...
                         AdaptBool( Output_GetSegmentLocalRotationHelical.Occluded ) );
      
      % Get the local segment rotation as a matrix
      Output_GetSegmentLocalRotationMatrix = MyClient.GetSegmentLocalRotationMatrix( SubjectName, SegmentName );
      fprintf( '        Local Rotation Matrix: (%g, %g, %g, %g, %g, %g, %g, %g, %g) %s\n', ...
                         Output_GetSegmentLocalRotationMatrix.Rotation( 1 ),               ...
                         Output_GetSegmentLocalRotationMatrix.Rotation( 2 ),               ...
                         Output_GetSegmentLocalRotationMatrix.Rotation( 3 ),               ...
                         Output_GetSegmentLocalRotationMatrix.Rotation( 4 ),               ...
                         Output_GetSegmentLocalRotationMatrix.Rotation( 5 ),               ...
                         Output_GetSegmentLocalRotationMatrix.Rotation( 6 ),               ...
                         Output_GetSegmentLocalRotationMatrix.Rotation( 7 ),               ...
                         Output_GetSegmentLocalRotationMatrix.Rotation( 8 ),               ...
                         Output_GetSegmentLocalRotationMatrix.Rotation( 9 ),               ...
                         AdaptBool( Output_GetSegmentLocalRotationMatrix.Occluded ) );
      
      % Get the local segment rotation in quaternion co-ordinates
      Output_GetSegmentLocalRotationQuaternion = MyClient.GetSegmentLocalRotationQuaternion( SubjectName, SegmentName );
      fprintf( '        Local Rotation Quaternion: (%g, %g, %g, %g) %s\n',             ...
                         Output_GetSegmentLocalRotationQuaternion.Rotation( 1 ),       ...
                         Output_GetSegmentLocalRotationQuaternion.Rotation( 2 ),       ...
                         Output_GetSegmentLocalRotationQuaternion.Rotation( 3 ),       ...
                         Output_GetSegmentLocalRotationQuaternion.Rotation( 4 ),       ...
                         AdaptBool( Output_GetSegmentLocalRotationQuaternion.Occluded ) );

      % Get the local segment rotation in EulerXYZ co-ordinates
      Output_GetSegmentLocalRotationEulerXYZ = MyClient.GetSegmentLocalRotationEulerXYZ( SubjectName, SegmentName );
      fprintf( '        Local Rotation EulerXYZ: (%g, %g, %g) %s\n',                 ...
                         Output_GetSegmentLocalRotationEulerXYZ.Rotation( 1 ),       ...
                         Output_GetSegmentLocalRotationEulerXYZ.Rotation( 2 ),       ...
                         Output_GetSegmentLocalRotationEulerXYZ.Rotation( 3 ),       ...
                         AdaptBool( Output_GetSegmentLocalRotationEulerXYZ.Occluded ) );

    end% SegmentIndex
    
    % Get the quality of the subject (object) if supported
    Output_GetObjectQuality = MyClient.GetObjectQuality( SubjectName );
    if Output_GetObjectQuality.Result == ViconDataStreamSDK.DotNET.Result.Success
      fprintf( '    Quality: %g\n', Output_GetObjectQuality.Quality );
    end
    
    % Count the number of markers
    MarkerCount = MyClient.GetMarkerCount( SubjectName ).MarkerCount;
    fprintf( '    Markers (%d):\n', MarkerCount );
    for MarkerIndex = 0:typecast( MarkerCount, 'int32' )-1
      % Get the marker name
      MarkerName = MyClient.GetMarkerName( SubjectName, typecast( MarkerIndex,'uint32') ).MarkerName;

      % Get the marker parent
      MarkerParentName = MyClient.GetMarkerParentName( SubjectName, MarkerName ).SegmentName;

      % Get the global marker translation
      Output_GetMarkerGlobalTranslation = MyClient.GetMarkerGlobalTranslation( SubjectName, MarkerName );

      fprintf( '      Marker #%d: %s (%g, %g, %g) %s\n',                     ...
                         MarkerIndex,                                    ...
                         char( MarkerName ),                                         ...
                         Output_GetMarkerGlobalTranslation.Translation( 1 ), ...
                         Output_GetMarkerGlobalTranslation.Translation( 2 ), ...
                         Output_GetMarkerGlobalTranslation.Translation( 3 ), ...
                         AdaptBool( Output_GetMarkerGlobalTranslation.Occluded ) );

      if bReadRays
        % Get the ray contributions for this marker
        Output_GetMarkerRayContributionCount = MyClient.GetMarkerRayContributionCount( SubjectName, MarkerName );
        if( Output_GetMarkerRayContributionCount.Result == ViconDataStreamSDK.DotNET.Result.Success )
          fprintf('      Contributed to by: ');

		  MarkerRayContributionCount = Output_GetMarkerRayContributionCount.RayContributionsCount;
          for ContributionIndex = 0: typecast( MarkerRayContributionCount, 'int32' )-1
            Output_GetMarkerRayContribution = MyClient.GetMarkerRayContribution(SubjectName, MarkerName, typecast( ContributionIndex, 'uint32' ) );
            fprintf( 'ID:%d Index:%d ', Output_GetMarkerRayContribution.CameraID, Output_GetMarkerRayContribution.CentroidIndex);
          end

          fprintf('\n' );
        end
      end% bReadRays
    end% MarkerIndex
    
  end% SubjectIndex

  % Get the unlabeled markers
  UnlabeledMarkerCount = MyClient.GetUnlabeledMarkerCount().MarkerCount;
  fprintf( '  Unlabeled Markers (%d):\n', UnlabeledMarkerCount );
  for UnlabeledMarkerIndex = 0:typecast( UnlabeledMarkerCount , 'int32' )- 1
    % Get the global marker translation
    Output_GetUnlabeledMarkerGlobalTranslation = MyClient.GetUnlabeledMarkerGlobalTranslation( typecast(UnlabeledMarkerIndex,'uint32') );
    if bTrajectoryIDs
      fprintf( '    Marker #%d: (%g, %g, %g): Traj ID %d\n',                                    ...
                         UnlabeledMarkerIndex,                                    ...
                         Output_GetUnlabeledMarkerGlobalTranslation.Translation( 1 ), ...
                         Output_GetUnlabeledMarkerGlobalTranslation.Translation( 2 ), ...
                         Output_GetUnlabeledMarkerGlobalTranslation.Translation( 3 ), ...
                         Output_GetUnlabeledMarkerGlobalTranslation.MarkerID );
    else
      fprintf( '    Marker #%d: (%g, %g, %g)\n',                                    ...
                         UnlabeledMarkerIndex,                                    ...
                         Output_GetUnlabeledMarkerGlobalTranslation.Translation( 1 ), ...
                         Output_GetUnlabeledMarkerGlobalTranslation.Translation( 2 ), ...
                         Output_GetUnlabeledMarkerGlobalTranslation.Translation( 3 ) );
    end 
  end% UnlabeledMarkerIndex

  % Get the labeled markers
  LabeledMarkerCount = MyClient.GetLabeledMarkerCount().MarkerCount;
  fprintf( '  Labeled Markers (%d):\n', LabeledMarkerCount );
  for LabeledMarkerIndex = 0:typecast( LabeledMarkerCount, 'int32' ) -1
    % Get the global marker translation
    Output_GetLabeledMarkerGlobalTranslation = MyClient.GetLabeledMarkerGlobalTranslation( typecast( LabeledMarkerIndex,'uint32') );
    if bTrajectoryIDs
      fprintf( '    Marker #%d: (%g, %g, %g): Traj ID %d\n',                                  ...
                         LabeledMarkerIndex,                                    ...
                         Output_GetLabeledMarkerGlobalTranslation.Translation( 1 ), ...
                         Output_GetLabeledMarkerGlobalTranslation.Translation( 2 ), ...
                         Output_GetLabeledMarkerGlobalTranslation.Translation( 3 ), ...
                         Output_GetLabeledMarkerGlobalTranslation.MarkerID );
    else
      fprintf( '    Marker #%d: (%g, %g, %g)\n',                                  ...
                         LabeledMarkerIndex,                                    ...
                         Output_GetLabeledMarkerGlobalTranslation.Translation( 1 ), ...
                         Output_GetLabeledMarkerGlobalTranslation.Translation( 2 ), ...
                         Output_GetLabeledMarkerGlobalTranslation.Translation( 3 ) );
    end
  end% LabeledMarkerIndex

  % Count the number of devices
  DeviceCount = MyClient.GetDeviceCount().DeviceCount;
  fprintf( '  Devices (%d):\n', DeviceCount );

  for DeviceIndex = 0:typecast( DeviceCount, 'int32' ) - 1
  
    fprintf( '    Device #%d:\n', DeviceIndex );
    
    % Get the device name and type
    Output_GetDeviceName = MyClient.GetDeviceName( typecast( DeviceIndex, 'uint32' ) );
    fprintf( '      Name: %s\n', char( Output_GetDeviceName.DeviceName ) );
    fprintf( '      Type: %s\n', char( Output_GetDeviceName.DeviceType.ToString() ) );

    % Count the number of device outputs
    DeviceOutputCount = MyClient.GetDeviceOutputCount( Output_GetDeviceName.DeviceName ).DeviceOutputCount;
    fprintf( '      Device Outputs (%d):\n', DeviceOutputCount );
    for DeviceOutputIndex = 0:typecast( DeviceOutputCount, 'int32' ) - 1
      % Get the device output name and unit
      Output_GetDeviceOutputComponentName = MyClient.GetDeviceOutputComponentName( Output_GetDeviceName.DeviceName, typecast( DeviceOutputIndex, 'uint32') );

      % Get the number of subsamples associated with this device.
      Output_GetDeviceOutputSubsamples = MyClient.GetDeviceOutputSubsamples( Output_GetDeviceName.DeviceName, Output_GetDeviceOutputComponentName.DeviceOutputName, Output_GetDeviceOutputComponentName.DeviceOutputComponentName ).DeviceOutputSubsamples;

      fprintf( '      Device Output #%d:\n', DeviceOutputIndex  );

      fprintf( '      Samples (%d):\n', Output_GetDeviceOutputSubsamples )
      for DeviceOutputSubsample = 0:typecast( Output_GetDeviceOutputSubsamples, 'int32' )-1 
          
        fprintf( '        Sample #%d:\n', DeviceOutputSubsample )  
        % Get the device output value
        Output_GetDeviceOutputValue = MyClient.GetDeviceOutputValue( Output_GetDeviceName.DeviceName, Output_GetDeviceOutputComponentName.DeviceOutputName, Output_GetDeviceOutputComponentName.DeviceOutputComponentName, typecast( DeviceOutputSubsample, 'uint32' ) );

        fprintf( '          ''%s'' %g %s %s\n',                                    ...
                           char( Output_GetDeviceOutputComponentName.DeviceOutputName ),            ...
                           Output_GetDeviceOutputValue.Value,                      ...
                           char( Output_GetDeviceOutputComponentName.DeviceOutputUnit.ToString() ), ...
                           AdaptBool( Output_GetDeviceOutputValue.Occluded ) );
      end% DeviceOutputSubsample
    end% DeviceOutputIndex
    
  end% DeviceIndex
    
  % Count the number of force plates
  ForcePlateCount = MyClient.GetForcePlateCount().ForcePlateCount;
  fprintf( '  Force Plates: (%d)\n', ForcePlateCount  );
  for ForcePlateIndex = 0:typecast( ForcePlateCount, 'int32' ) -1
    fprintf( '    Force Plate #%d:\n', ForcePlateIndex  );
    
    % Get the number of subsamples associated with this device.
    Output_GetForcePlateSubsamples = MyClient.GetForcePlateSubsamples( typecast( ForcePlateIndex, 'uint32') );
    
    fprintf( '    Samples (%d):\n', Output_GetForcePlateSubsamples.ForcePlateSubsamples );
    for ForcePlateSubsample = 0:typecast( Output_GetForcePlateSubsamples.ForcePlateSubsamples, 'int32' ) - 1
      % Output all the subsamples.
      fprintf( '      Sample #%d:\n', ForcePlateSubsample );
      
      Output_GetGlobalForceVector = MyClient.GetGlobalForceVector( typecast( ForcePlateIndex, 'uint32' ), typecast( ForcePlateSubsample, 'uint32') );
      fprintf( '      Force (%g, %g, %g)\n',                           ...
                         Output_GetGlobalForceVector.ForceVector( 1 ), ...
                         Output_GetGlobalForceVector.ForceVector( 2 ), ...
                         Output_GetGlobalForceVector.ForceVector( 3 ) );
    
      Output_GetGlobalMomentVector = MyClient.GetGlobalMomentVector( typecast( ForcePlateIndex, 'uint32'), typecast( ForcePlateSubsample, 'uint32') );
      fprintf( '      Moment (%g, %g, %g)\n',                            ...
                         Output_GetGlobalMomentVector.MomentVector( 1 ), ...
                         Output_GetGlobalMomentVector.MomentVector( 2 ), ...
                         Output_GetGlobalMomentVector.MomentVector( 3 ) );
    
      Output_GetGlobalCentreOfPressure = MyClient.GetGlobalCentreOfPressure( typecast( ForcePlateIndex, 'uint32' ), typecast( ForcePlateSubsample, 'uint32') );
      fprintf( '      CoP (%g, %g, %g)\n',                                       ...
                         Output_GetGlobalCentreOfPressure.CentreOfPressure( 1 ), ...
                         Output_GetGlobalCentreOfPressure.CentreOfPressure( 2 ), ...
                         Output_GetGlobalCentreOfPressure.CentreOfPressure( 3 ) );    
    end% ForcePlateSubsample                     
  end% ForcePlateIndex
  
  % Count the number of eye trackers
  EyeTrackerCount = MyClient.GetEyeTrackerCount().EyeTrackerCount;
  fprintf( '  Eye Trackers: (%d)\n', EyeTrackerCount );
  for EyeTrackerIndex = 0:typecast( EyeTrackerCount, 'int32' ) - 1
    fprintf( '    Eye Tracker #%d:\n', EyeTrackerIndex );

    Output_GetEyeTrackerGlobalPosition = MyClient.GetEyeTrackerGlobalPosition( typecast( EyeTrackerIndex, 'uint32') );
    fprintf( '      Position (%g, %g, %g) %s\n',                       ...
                     Output_GetEyeTrackerGlobalPosition.Position( 1 ), ...
                     Output_GetEyeTrackerGlobalPosition.Position( 2 ), ...
                     Output_GetEyeTrackerGlobalPosition.Position( 3 ), ...
                     AdaptBool( Output_GetEyeTrackerGlobalPosition.Occluded ) );
   
    Output_GetEyeTrackerGlobalGazeVector = MyClient.GetEyeTrackerGlobalGazeVector( typecast( EyeTrackerIndex, 'uint32') );
    fprintf( '      Gaze (%g, %g, %g) %s\n',                               ...
                     Output_GetEyeTrackerGlobalGazeVector.GazeVector( 1 ), ...
                     Output_GetEyeTrackerGlobalGazeVector.GazeVector( 2 ), ...
                     Output_GetEyeTrackerGlobalGazeVector.GazeVector( 3 ), ... 
                     AdaptBool( Output_GetEyeTrackerGlobalGazeVector.Occluded ) );
     
  end% EyeTrackerIndex  
    
  if bReadCentroids
    CameraCount = MyClient.GetCameraCount().CameraCount;
    fprintf( 'Cameras(%d):\n', CameraCount);

    for CameraIndex = 0:typecast( CameraCount, 'int32' ) - 1
      fprintf('  Camera #%d:\n', typecast( CameraIndex, 'uint32' ) );
        
      CameraName = MyClient.GetCameraName( typecast( CameraIndex, 'uint32' ) ).CameraName;
      fprintf ( '    Name: %s\n', char( CameraName ) );

      CentroidCount = MyClient.GetCentroidCount( CameraName ).CentroidCount;
      fprintf ( '    Centroids(%d):\n', CentroidCount );

      for CentroidIndex = 0:typecast( CentroidCount, 'int32' ) - 1
        fprintf( '      Centroid #%d:\n', CentroidIndex  );

        Output_GetCentroidPosition = MyClient.GetCentroidPosition( CameraName, typecast( CentroidIndex, 'uint32') );
        fprintf( '        Position: (%g, %g)\n', Output_GetCentroidPosition.CentroidPosition( 1 ), ...
                                                 Output_GetCentroidPosition.CentroidPosition( 2 ) );
        fprintf( '        Radius: (%g)\n', Output_GetCentroidPosition.Radius );
        %fprintf( '        Accuracy: (%g)\n', Output_GetCentroidPosition.Accuracy );
      end% CentroidIndex
    end% CameraIndex
  end% bReadCentroids

end% while true  
tEnd = toc( tStart );
fprintf( 'Time Elapsed: %d\n', tEnd );

if TransmitMulticast
  MyClient.StopTransmittingMulticast();
end  

% Disconnect and dispose
MyClient.Disconnect();

% % Unload the SDK
% fprintf( 'Unloading SDK...' );
% Client.UnloadViconDataStreamSDK();
% fprintf( 'done\n' );

%fprintf( 'Skipped %d frames\n', size(SkippedFrames,2) );


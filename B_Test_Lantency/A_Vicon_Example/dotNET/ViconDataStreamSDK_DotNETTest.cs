///////////////////////////////////////////////////////////////////////////////
//
// Copyright (C) OMG Plc 2009.
// All rights reserved.  This software is protected by copyright
// law and international treaties.  No part of this software / document
// may be reproduced or distributed in any form or by any means,
// whether transiently or incidentally to some other use of this software,
// without the written permission of the copyright owner.
//
///////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Generic;
using System.Text;
using ViconDataStreamSDK.DotNET;

namespace CSharpClient
{
  class Program
  {
    static string Adapt( TimecodeStandard i_Standard )
    {
      switch( i_Standard )
      {
        default:
        case TimecodeStandard.None:
          return "0";
        case TimecodeStandard.PAL:
          return "1";
        case TimecodeStandard.NTSC:
          return "2";
        case TimecodeStandard.NTSCDrop:
          return "3";
        case TimecodeStandard.Film:
          return "4";
        case TimecodeStandard.NTSCFilm:
          return "5";
        case TimecodeStandard.ATSC:
          return "6";
      }
    }

    static string Adapt( Direction i_Direction )
    {
      switch( i_Direction )
      {
        case Direction.Forward:
          return "Forward";
        case Direction.Backward:
          return "Backward";
        case Direction.Left:
          return "Left";
        case Direction.Right:
          return "Right";
        case Direction.Up:
          return "Up";
        case Direction.Down:
          return "Down";
        default:
          return "Unknown";
      }
    }

    static string Adapt( DeviceType i_DeviceType )
    {
      switch( i_DeviceType )
      {
        case DeviceType.ForcePlate:
          return "ForcePlate";
        case DeviceType.Unknown:
        default:
          return "Unknown";
      }
    }

    static string Adapt( Unit i_Unit )
    {
      switch( i_Unit )
      {
        case Unit.Meter:
          return "Meter";
        case Unit.Volt:
          return "Volt";
        case Unit.NewtonMeter:
          return "NewtonMeter";
        case Unit.Newton:
          return "Newton";
        case Unit.Kilogram:
          return "Kilogram";
        case Unit.Second:
          return "Second";
        case Unit.Ampere:
          return "Ampere";
        case Unit.Kelvin:
          return "Kelvin";
        case Unit.Mole:
          return "Mole";
        case Unit.Candela:
          return "Candela";
        case Unit.Radian:
          return "Radian";
        case Unit.Steradian:
          return "Steradian";
        case Unit.MeterSquared:
          return "MeterSquared";
        case Unit.MeterCubed:
          return "MeterCubed";
        case Unit.MeterPerSecond:
          return "MeterPerSecond";
        case Unit.MeterPerSecondSquared:
          return "MeterPerSecondSquared";
        case Unit.RadianPerSecond:
          return "RadianPerSecond";
        case Unit.RadianPerSecondSquared:
          return "RadianPerSecondSquared";
        case Unit.Hertz:
          return "Hertz";
        case Unit.Joule:
          return "Joule";
        case Unit.Watt:
          return "Watt";
        case Unit.Pascal:
          return "Pascal";
        case Unit.Lumen:
          return "Lumen";
        case Unit.Lux:
          return "Lux";
        case Unit.Coulomb:
          return "Coulomb";
        case Unit.Ohm:
          return "Ohm";
        case Unit.Farad:
          return "Farad";
        case Unit.Weber:
          return "Weber";
        case Unit.Tesla:
          return "Tesla";
        case Unit.Henry:
          return "Henry";
        case Unit.Siemens:
          return "Siemens";
        case Unit.Becquerel:
          return "Becquerel";
        case Unit.Gray:
          return "Gray";
        case Unit.Sievert:
          return "Sievert";
        case Unit.Katal:
          return "Katal";

        case Unit.Unknown:
        default:
          return "Unknown";
      }
    }

    static void Main( string[] args )
    {
      // Program options
      bool TransmitMulticast = false;
      bool EnableHapticTest = false;
      bool bReadCentroids = false;
      bool bReadRayData = false;
      bool bReadGreyscaleData = false;
      bool bReadVideoData = false;
      bool bMarkerTrajIds = false;
      bool bLightweightSegments = false;
      bool bSubjectFilterApplied = false;

      List<String> HapticOnList = new List<String>();
      uint ClientBufferSize = 0;
      string AxisMapping = "ZUp";
      List<String> FilteredSubjects = new List<string>();
      

      int Counter = 1;
      

      string HostName = "localhost:801";
      if( args.Length > 0 )
      {
        HostName = args[0];
      }

      // parsing all the haptic arguments
      for ( int i = 1; i < args.Length; ++i )
      {
        string HapticArg = "--enable_haptic_test";
        if ( String.Compare( args[i], HapticArg ) == 0 )
        {
          EnableHapticTest = true;
          ++i;
          while( i < args.Length && String.Compare( args[i],0, "--", 0, 2 ) != 0  )
          {
            HapticOnList.Add( args[i]);
            ++i;
          }
        }

        string CentroidsArg = "--centroids";
        if (String.Compare(args[i], CentroidsArg) == 0)
        {
          bReadCentroids = true;
        }

        string RaysArg = "--rays";
        if (String.Compare(args[i], RaysArg) == 0)
        {
            bReadRayData = true;
        }

        string GreyscaleArg = "--greyscale";
        if (String.Compare(args[i], GreyscaleArg) == 0)
        {
          bReadGreyscaleData = true;
          bReadCentroids = true;
          Console.WriteLine("Enabling greyscale data also enables centroid output");
        }

        string VideoArg = "--video";
        if (String.Compare(args[i], VideoArg) == 0)
        {
          bReadVideoData = true;
          bReadCentroids = true;
          Console.WriteLine("Enabling video data also enables centroid output");
        }

        string MarkerTrajIdsArg = "--marker-traj-id";
        if( String.Compare(args[i], MarkerTrajIdsArg) == 0)
        {
          bMarkerTrajIds = true;
        }

        string BufferSizeArg = "--client-buffer-size";
        if( String.Compare( args[i], BufferSizeArg ) == 0 )
        {
          ++i;
          if( i < args.Length )
          {
            try
            {
              ClientBufferSize = Convert.ToUInt32(args[i]);
            }
            catch( FormatException )
            {
              Console.WriteLine( "Invalid argument for --client-buffer-size" );
            }
            catch( OverflowException )
            {
              Console.WriteLine( "Invalid argument for --client-buffer-size" );
            }
          }
        }
        
        string SetAxisMapping = "--set-axis-mapping";
        if (String.Compare(args[i], SetAxisMapping) == 0)
        { 
          ++i;
          AxisMapping = args[i];
          if ( String.Compare( AxisMapping, "XUp") == 0
            || String.Compare( AxisMapping, "YUp") == 0
            || String.Compare( AxisMapping, "ZUp") == 0 )
          { 
              Console.WriteLine( "Setting Axis to {0}", AxisMapping );
          }
          else
          {
              Console.WriteLine( "Invalid Axis setting: {0}. Should be XUp, YUp or ZUp", AxisMapping );
              AxisMapping = "ZUp";
          }
        }

        string LightweightSegments = "--lightweight";
        if( String.Compare( args[i], LightweightSegments ) == 0 )
        {
          bLightweightSegments = true;
        }

        string SubjectFilter = "--subjects";
        if (String.Compare(args[i], SubjectFilter) == 0)
        {
          ++i;
          while (i < args.Length && String.Compare( args[i], 0, "--", 0, 2) != 0)
          {
            FilteredSubjects.Add( args[i] );
            ++i;
          }
        }
      }

      // Make a new client
      ViconDataStreamSDK.DotNET.Client MyClient = new ViconDataStreamSDK.DotNET.Client();

      // Connect to a server
      Console.Write( "Connecting to {0} ...", HostName );
      while( !MyClient.IsConnected().Connected )
      {
        // Direct connection
        MyClient.Connect( HostName );

        // Multicast connection
        // MyClient.ConnectToMulticast( HostName, "224.0.0.0" );

        System.Threading.Thread.Sleep( 200 );
        Console.Write( "." );
      }
      Console.WriteLine();

      // Enable some different data types
      MyClient.EnableSegmentData();
      MyClient.EnableMarkerData();
      MyClient.EnableUnlabeledMarkerData();
      MyClient.EnableDeviceData();
      if (bReadCentroids)
      {
        MyClient.EnableCentroidData();
      }
      if( bReadRayData )
      {
        MyClient.EnableMarkerRayData();
      }
      if( bReadGreyscaleData)
      {
        MyClient.EnableGreyscaleData();
      }
      if( bReadVideoData)
      {
        MyClient.EnableVideoData();
      }
      if( bLightweightSegments )
      {
        MyClient.EnableLightweightSegmentData();
      }

      Console.WriteLine( "Segment Data Enabled: {0}",          MyClient.IsSegmentDataEnabled().Enabled );
      Console.WriteLine( "Lightweight Segment Data Enabled: {0}", MyClient.IsLightweightSegmentDataEnabled().Enabled);
      Console.WriteLine( "Marker Data Enabled: {0}",           MyClient.IsMarkerDataEnabled().Enabled );
      Console.WriteLine( "Unlabeled Marker Data Enabled: {0}", MyClient.IsUnlabeledMarkerDataEnabled().Enabled );
      Console.WriteLine( "Device Data Enabled: {0}",           MyClient.IsDeviceDataEnabled().Enabled );
      Console.WriteLine( "Centroid Data Enabled: {0}",         MyClient.IsCentroidDataEnabled().Enabled );
      Console.WriteLine( "Marker Ray Data Enabled: {0}",       MyClient.IsMarkerRayDataEnabled().Enabled);
      Console.WriteLine( "Greyscale Data Enabled: {0}",        MyClient.IsGreyscaleDataEnabled().Enabled);
      Console.WriteLine( "Video Data Enabled: {0}",            MyClient.IsVideoDataEnabled().Enabled);
      Console.WriteLine( "Debug Data Enabled: {0}",            MyClient.IsDebugDataEnabled().Enabled);

      // Set the streaming mode
      MyClient.SetStreamMode( ViconDataStreamSDK.DotNET.StreamMode.ClientPull );
      // MyClient.SetStreamMode( ViconDataStreamSDK.DotNET.StreamMode.ClientPullPreFetch );
      // MyClient.SetStreamMode( ViconDataStreamSDK.DotNET.StreamMode.ServerPush );

      // Set the global up axis
      MyClient.SetAxisMapping( ViconDataStreamSDK.DotNET.Direction.Forward,
                               ViconDataStreamSDK.DotNET.Direction.Left,
                               ViconDataStreamSDK.DotNET.Direction.Up ); // Z-up

      if( AxisMapping == "YUp")
      { 
          MyClient.SetAxisMapping(ViconDataStreamSDK.DotNET.Direction.Forward,
                                   ViconDataStreamSDK.DotNET.Direction.Up,
                                   ViconDataStreamSDK.DotNET.Direction.Right); // Y-up
      }
      else if( AxisMapping == "XUp")
      { 
          MyClient.SetAxisMapping(ViconDataStreamSDK.DotNET.Direction.Up,
                                   ViconDataStreamSDK.DotNET.Direction.Forward,
                                   ViconDataStreamSDK.DotNET.Direction.Left ); // x-up
      }


      Output_GetAxisMapping _Output_GetAxisMapping = MyClient.GetAxisMapping();
      Console.WriteLine( "Axis Mapping: X-{0} Y-{1} Z-{2}", Adapt( _Output_GetAxisMapping.XAxis ),
                                                            Adapt( _Output_GetAxisMapping.YAxis ),
                                                            Adapt( _Output_GetAxisMapping.ZAxis ) );

      // Discover the version number
      Output_GetVersion _Output_GetVersion = MyClient.GetVersion();
      Console.WriteLine( "Version: {0}.{1}.{2}.{3}", _Output_GetVersion.Major, 
                                                 _Output_GetVersion.Minor, 
                                                 _Output_GetVersion.Point,
                                                 _Output_GetVersion.Revision );

      if( ClientBufferSize > 0 )
      {
        MyClient.SetBufferSize( ClientBufferSize );
        Console.WriteLine( "Setting client buffer size to " + ClientBufferSize );
      }

      if( TransmitMulticast )
      {
        MyClient.StartTransmittingMulticast( "localhost", "224.0.0.0" );
      }

      // Loop until a key is pressed
      while( true )
      {
        ++Counter;
        // Console.KeyAvailable throws an exception if stdin is a pipe (e.g.
        // with TrackerDssdkTests.py), so we use try..catch:
        try
        {
          if ( Console.KeyAvailable)
          {
            break;
          }
        }
        catch( InvalidOperationException)
        {
        }
        
        // Get a frame
        Console.Write( "Waiting for new frame..." );
        while( MyClient.GetFrame().Result != ViconDataStreamSDK.DotNET.Result.Success )
        {
          System.Threading.Thread.Sleep(200);
          Console.Write( "." );
        }
        Console.WriteLine();

        if (!bSubjectFilterApplied)
        {
          MyClient.ClearSubjectFilter(); 
          foreach (String Subject in FilteredSubjects)
          {
            Output_AddToSubjectFilter Result = MyClient.AddToSubjectFilter(Subject);
            bSubjectFilterApplied = bSubjectFilterApplied || Result.Result == ViconDataStreamSDK.DotNET.Result.Success;
          }
        }

        // Get the frame number
        Output_GetFrameNumber _Output_GetFrameNumber = MyClient.GetFrameNumber();
        Console.WriteLine( "Frame Number: {0}", _Output_GetFrameNumber.FrameNumber );

        Output_GetFrameRate _Output_GetFrameRate = MyClient.GetFrameRate();
        Console.WriteLine("Frame rate: {0}", _Output_GetFrameRate.FrameRateHz);


        for( uint FrameRateIndex = 0 ; FrameRateIndex < MyClient.GetFrameRateCount().Count ; ++FrameRateIndex )
        {
          string FrameRateName  = MyClient.GetFrameRateName( FrameRateIndex ).Name;
          double FrameRateValue = MyClient.GetFrameRateValue( FrameRateName ).Value;

          Console.WriteLine( "{0}: {1}Hz", FrameRateName, FrameRateValue );
        }
        Console.WriteLine();

        // Get the timecode
        Output_GetTimecode _Output_GetTimecode  = MyClient.GetTimecode();
        Console.WriteLine( "Timecode: {0}h {1}m {2}s {3}f {4}sf {5} {6} {7} {8}",
                           _Output_GetTimecode.Hours,
                           _Output_GetTimecode.Minutes,
                           _Output_GetTimecode.Seconds,
                           _Output_GetTimecode.Frames,
                           _Output_GetTimecode.SubFrame,
                           _Output_GetTimecode.FieldFlag,
                           Adapt( _Output_GetTimecode.Standard ),
                           _Output_GetTimecode.SubFramesPerFrame,
                           _Output_GetTimecode.UserBits );
        Console.WriteLine();

        // Get the latency
        Console.WriteLine( "Latency: {0}s", MyClient.GetLatencyTotal().Total );
        
        for( uint LatencySampleIndex = 0 ; LatencySampleIndex < MyClient.GetLatencySampleCount().Count ; ++LatencySampleIndex )
        {
          string SampleName  = MyClient.GetLatencySampleName( LatencySampleIndex ).Name;
          double SampleValue = MyClient.GetLatencySampleValue( SampleName ).Value;

          Console.WriteLine( "  {0} {1}s", SampleName, SampleValue );
        }
        Console.WriteLine();

        Output_GetHardwareFrameNumber _Output_GetHardwareFrameNumber = MyClient.GetHardwareFrameNumber();
        Console.WriteLine( "Hardware Frame Number: {0} ", _Output_GetHardwareFrameNumber.HardwareFrameNumber );

        //Enable haptic devices
        if ( EnableHapticTest == true )
        {
          foreach( String HapticDevice in HapticOnList)
          {
            if( Counter % 2 == 0 )
            {
                Output_SetApexDeviceFeedback Output= MyClient.SetApexDeviceFeedback( HapticDevice,  true ); 
                if( Output.Result == ViconDataStreamSDK.DotNET.Result.Success )
                {
                  Console.WriteLine( "Turn haptic feedback on for device: {0}\n", HapticDevice);
                }
                else if( Output.Result == ViconDataStreamSDK.DotNET.Result.InvalidDeviceName )
                {
                  Console.WriteLine( "Device doesn't exis: {0}\n", HapticDevice);
                }
            }
            if( Counter % 20 == 0 )
            {
                Output_SetApexDeviceFeedback Output = MyClient.SetApexDeviceFeedback( HapticDevice,  false);

                if (Output.Result == ViconDataStreamSDK.DotNET.Result.Success)
                {
                  Console.WriteLine( "Turn haptic feedback off for device: {0}\n", HapticDevice);
                }
            }
          }
        }
        // Count the number of subjects
        uint SubjectCount = MyClient.GetSubjectCount().SubjectCount;
        Console.WriteLine( "Subjects ({0}):", SubjectCount );
        for( uint SubjectIndex = 0 ; SubjectIndex < SubjectCount ; ++SubjectIndex )
        {
          Console.WriteLine( "  Subject #{0}", SubjectIndex );

          // Get the subject name
          string SubjectName = MyClient.GetSubjectName( SubjectIndex ).SubjectName;
          Console.WriteLine( "    Name: {0}", SubjectName );

          // Get the root segment
          string RootSegment = MyClient.GetSubjectRootSegmentName( SubjectName ).SegmentName;
          Console.WriteLine( "    Root Segment: {0}", RootSegment );

          // Count the number of segments
          uint SegmentCount = MyClient.GetSegmentCount( SubjectName ).SegmentCount;
          Console.WriteLine( "    Segments ({0}):", SegmentCount );
          for( uint SegmentIndex = 0 ; SegmentIndex < SegmentCount ; ++SegmentIndex )
          {
            Console.WriteLine( "      Segment #{0}", SegmentIndex );

            // Get the segment name
            string SegmentName = MyClient.GetSegmentName( SubjectName, SegmentIndex ).SegmentName;
            Console.WriteLine( "        Name: {0}", SegmentName );

            // Get the segment parent
            string SegmentParentName = MyClient.GetSegmentParentName( SubjectName, SegmentName ).SegmentName;
            Console.WriteLine( "        Parent: {0}",  SegmentParentName );

            // Get the segment's children
            uint ChildCount = MyClient.GetSegmentChildCount( SubjectName, SegmentName ).SegmentCount;
            Console.WriteLine( "     Children ({0}):", ChildCount );
            for( uint ChildIndex = 0 ; ChildIndex < ChildCount ; ++ChildIndex )
            {
              string ChildName = MyClient.GetSegmentChildName( SubjectName, SegmentName, ChildIndex ).SegmentName;
              Console.WriteLine( "       {0}", ChildName );
            }

            // Get the static segment translation
            Output_GetSegmentStaticTranslation _Output_GetSegmentStaticTranslation =
              MyClient.GetSegmentStaticTranslation(SubjectName, SegmentName);
            Console.WriteLine("        Static Translation: ({0},{1},{2})",
                               _Output_GetSegmentStaticTranslation.Translation[0],
                               _Output_GetSegmentStaticTranslation.Translation[1],
                               _Output_GetSegmentStaticTranslation.Translation[2]);

            // Get the static segment rotation in helical co-ordinates
            Output_GetSegmentStaticRotationHelical _Output_GetSegmentStaticRotationHelical =
              MyClient.GetSegmentStaticRotationHelical(SubjectName, SegmentName);
            Console.WriteLine("        Static Rotation Helical: ({0},{1},{2})",
                               _Output_GetSegmentStaticRotationHelical.Rotation[0],
                               _Output_GetSegmentStaticRotationHelical.Rotation[1],
                               _Output_GetSegmentStaticRotationHelical.Rotation[2]);

            // Get the static segment rotation as a matrix
            Output_GetSegmentStaticRotationMatrix _Output_GetSegmentStaticRotationMatrix =
              MyClient.GetSegmentStaticRotationMatrix(SubjectName, SegmentName);
            Console.WriteLine("        Static Rotation Matrix: ({0},{1},{2},{3},{4},{5},{6},{7},{8})",
                               _Output_GetSegmentStaticRotationMatrix.Rotation[0],
                               _Output_GetSegmentStaticRotationMatrix.Rotation[1],
                               _Output_GetSegmentStaticRotationMatrix.Rotation[2],
                               _Output_GetSegmentStaticRotationMatrix.Rotation[3],
                               _Output_GetSegmentStaticRotationMatrix.Rotation[4],
                               _Output_GetSegmentStaticRotationMatrix.Rotation[5],
                               _Output_GetSegmentStaticRotationMatrix.Rotation[6],
                               _Output_GetSegmentStaticRotationMatrix.Rotation[7],
                               _Output_GetSegmentStaticRotationMatrix.Rotation[8]);

            // Get the static segment rotation in quaternion co-ordinates
            Output_GetSegmentStaticRotationQuaternion _Output_GetSegmentStaticRotationQuaternion =
              MyClient.GetSegmentStaticRotationQuaternion(SubjectName, SegmentName);
            Console.WriteLine("        Static Rotation Quaternion: ({0},{1},{2},{3})",
                               _Output_GetSegmentStaticRotationQuaternion.Rotation[0],
                               _Output_GetSegmentStaticRotationQuaternion.Rotation[1],
                               _Output_GetSegmentStaticRotationQuaternion.Rotation[2],
                               _Output_GetSegmentStaticRotationQuaternion.Rotation[3]);

            // Get the static segment rotation in EulerXYZ co-ordinates
            Output_GetSegmentStaticRotationEulerXYZ _Output_GetSegmentStaticRotationEulerXYZ =
              MyClient.GetSegmentStaticRotationEulerXYZ(SubjectName, SegmentName);
            Console.WriteLine("        Static Rotation EulerXYZ: ({0},{1},{2})",
                               _Output_GetSegmentStaticRotationEulerXYZ.Rotation[0],
                               _Output_GetSegmentStaticRotationEulerXYZ.Rotation[1],
                               _Output_GetSegmentStaticRotationEulerXYZ.Rotation[2]);

            // Get the global segment translation
            Output_GetSegmentGlobalTranslation _Output_GetSegmentGlobalTranslation = 
              MyClient.GetSegmentGlobalTranslation( SubjectName, SegmentName );
            Console.WriteLine( "        Global Translation: ({0},{1},{2}) {3}",
                               _Output_GetSegmentGlobalTranslation.Translation[ 0 ],
                               _Output_GetSegmentGlobalTranslation.Translation[ 1 ],
                               _Output_GetSegmentGlobalTranslation.Translation[ 2 ],
                               _Output_GetSegmentGlobalTranslation.Occluded );

            // Get the global segment rotation in helical co-ordinates
            Output_GetSegmentGlobalRotationHelical _Output_GetSegmentGlobalRotationHelical = 
              MyClient.GetSegmentGlobalRotationHelical( SubjectName, SegmentName );
            Console.WriteLine( "        Global Rotation Helical: ({0},{1},{2}) {3}",
                               _Output_GetSegmentGlobalRotationHelical.Rotation[ 0 ],
                               _Output_GetSegmentGlobalRotationHelical.Rotation[ 1 ],
                               _Output_GetSegmentGlobalRotationHelical.Rotation[ 2 ],
                               _Output_GetSegmentGlobalRotationHelical.Occluded );

            // Get the global segment rotation as a matrix
            Output_GetSegmentGlobalRotationMatrix _Output_GetSegmentGlobalRotationMatrix = 
              MyClient.GetSegmentGlobalRotationMatrix( SubjectName, SegmentName );
            Console.WriteLine( "        Global Rotation Matrix: ({0},{1},{2},{3},{4},{5},{6},{7},{8}) {9}",
                               _Output_GetSegmentGlobalRotationMatrix.Rotation[ 0 ],
                               _Output_GetSegmentGlobalRotationMatrix.Rotation[ 1 ],
                               _Output_GetSegmentGlobalRotationMatrix.Rotation[ 2 ],
                               _Output_GetSegmentGlobalRotationMatrix.Rotation[ 3 ],
                               _Output_GetSegmentGlobalRotationMatrix.Rotation[ 4 ],
                               _Output_GetSegmentGlobalRotationMatrix.Rotation[ 5 ],
                               _Output_GetSegmentGlobalRotationMatrix.Rotation[ 6 ],
                               _Output_GetSegmentGlobalRotationMatrix.Rotation[ 7 ],
                               _Output_GetSegmentGlobalRotationMatrix.Rotation[ 8 ],
                               _Output_GetSegmentGlobalRotationMatrix.Occluded );

            // Get the global segment rotation in quaternion co-ordinates
            Output_GetSegmentGlobalRotationQuaternion _Output_GetSegmentGlobalRotationQuaternion = 
              MyClient.GetSegmentGlobalRotationQuaternion( SubjectName, SegmentName );
            Console.WriteLine( "        Global Rotation Quaternion: ({0},{1},{2},{3}) {4}",
                               _Output_GetSegmentGlobalRotationQuaternion.Rotation[ 0 ],
                               _Output_GetSegmentGlobalRotationQuaternion.Rotation[ 1 ],
                               _Output_GetSegmentGlobalRotationQuaternion.Rotation[ 2 ],
                               _Output_GetSegmentGlobalRotationQuaternion.Rotation[ 3 ],
                               _Output_GetSegmentGlobalRotationQuaternion.Occluded );

            // Get the global segment rotation in EulerXYZ co-ordinates
            Output_GetSegmentGlobalRotationEulerXYZ _Output_GetSegmentGlobalRotationEulerXYZ = 
              MyClient.GetSegmentGlobalRotationEulerXYZ( SubjectName, SegmentName );
            Console.WriteLine( "        Global Rotation EulerXYZ: ({0},{1},{2}) {3}",
                               _Output_GetSegmentGlobalRotationEulerXYZ.Rotation[ 0 ],
                               _Output_GetSegmentGlobalRotationEulerXYZ.Rotation[ 1 ],
                               _Output_GetSegmentGlobalRotationEulerXYZ.Rotation[ 2 ],
                               _Output_GetSegmentGlobalRotationEulerXYZ.Occluded );

            // Get the local segment translation
            Output_GetSegmentLocalTranslation _Output_GetSegmentLocalTranslation = 
              MyClient.GetSegmentLocalTranslation( SubjectName, SegmentName );
            Console.WriteLine( "        Local Translation: ({0},{1},{2}) {3}",
                               _Output_GetSegmentLocalTranslation.Translation[ 0 ],
                               _Output_GetSegmentLocalTranslation.Translation[ 1 ],
                               _Output_GetSegmentLocalTranslation.Translation[ 2 ],
                               _Output_GetSegmentLocalTranslation.Occluded );

            // Get the local segment rotation in helical co-ordinates
            Output_GetSegmentLocalRotationHelical _Output_GetSegmentLocalRotationHelical = 
              MyClient.GetSegmentLocalRotationHelical( SubjectName, SegmentName );
            Console.WriteLine( "        Local Rotation Helical: ({0},{1},{2}) {3}",
                               _Output_GetSegmentLocalRotationHelical.Rotation[ 0 ],
                               _Output_GetSegmentLocalRotationHelical.Rotation[ 1 ],
                               _Output_GetSegmentLocalRotationHelical.Rotation[ 2 ],
                               _Output_GetSegmentLocalRotationHelical.Occluded );

            // Get the local segment rotation as a matrix
            Output_GetSegmentLocalRotationMatrix _Output_GetSegmentLocalRotationMatrix = 
              MyClient.GetSegmentLocalRotationMatrix( SubjectName, SegmentName );
            Console.WriteLine( "        Local Rotation Matrix: ({0},{1},{2},{3},{4},{5},{6},{7},{8}) {9}",
                               _Output_GetSegmentLocalRotationMatrix.Rotation[ 0 ],
                               _Output_GetSegmentLocalRotationMatrix.Rotation[ 1 ],
                               _Output_GetSegmentLocalRotationMatrix.Rotation[ 2 ],
                               _Output_GetSegmentLocalRotationMatrix.Rotation[ 3 ],
                               _Output_GetSegmentLocalRotationMatrix.Rotation[ 4 ],
                               _Output_GetSegmentLocalRotationMatrix.Rotation[ 5 ],
                               _Output_GetSegmentLocalRotationMatrix.Rotation[ 6 ],
                               _Output_GetSegmentLocalRotationMatrix.Rotation[ 7 ],
                               _Output_GetSegmentLocalRotationMatrix.Rotation[ 8 ],
                               _Output_GetSegmentLocalRotationMatrix.Occluded );

            // Get the local segment rotation in quaternion co-ordinates
            Output_GetSegmentLocalRotationQuaternion _Output_GetSegmentLocalRotationQuaternion = 
              MyClient.GetSegmentLocalRotationQuaternion( SubjectName, SegmentName );
            Console.WriteLine( "        Local Rotation Quaternion: ({0},{1},{2},{3}) {4}",
                               _Output_GetSegmentLocalRotationQuaternion.Rotation[ 0 ],
                               _Output_GetSegmentLocalRotationQuaternion.Rotation[ 1 ],
                               _Output_GetSegmentLocalRotationQuaternion.Rotation[ 2 ],
                               _Output_GetSegmentLocalRotationQuaternion.Rotation[ 3 ],
                               _Output_GetSegmentLocalRotationQuaternion.Occluded );

            // Get the local segment rotation in EulerXYZ co-ordinates
            Output_GetSegmentLocalRotationEulerXYZ _Output_GetSegmentLocalRotationEulerXYZ = 
              MyClient.GetSegmentLocalRotationEulerXYZ( SubjectName, SegmentName );
            Console.WriteLine( "        Local Rotation EulerXYZ: ({0},{1},{2}) {3}",
                               _Output_GetSegmentLocalRotationEulerXYZ.Rotation[ 0 ],
                               _Output_GetSegmentLocalRotationEulerXYZ.Rotation[ 1 ],
                               _Output_GetSegmentLocalRotationEulerXYZ.Rotation[ 2 ],
                               _Output_GetSegmentLocalRotationEulerXYZ.Occluded );
          }

          // Get the quality of the subject (object) if supported
          Output_GetObjectQuality _Output_GetObjectQuality = MyClient.GetObjectQuality( SubjectName );
          if( _Output_GetObjectQuality.Result == Result.Success )
          {
            double Quality = _Output_GetObjectQuality.Quality;
            Console.WriteLine( "    Quality: {0}", Quality );
          }

          // Count the number of markers
          uint MarkerCount = MyClient.GetMarkerCount( SubjectName ).MarkerCount;
          Console.WriteLine( "    Markers ({0}):", MarkerCount );
          for( uint MarkerIndex = 0 ; MarkerIndex < MarkerCount ; ++MarkerIndex )
          {
            // Get the marker name
            string MarkerName = MyClient.GetMarkerName( SubjectName, MarkerIndex ).MarkerName;

            // Get the marker parent
            string MarkerParentName = MyClient.GetMarkerParentName(SubjectName, MarkerName).SegmentName;

            // Get the global marker translation
            Output_GetMarkerGlobalTranslation _Output_GetMarkerGlobalTranslation = 
              MyClient.GetMarkerGlobalTranslation( SubjectName, MarkerName );

            Console.WriteLine( "      Marker #{0}: {1} ({2}, {3}, {4}) {5}", 
                               MarkerIndex, 
                               MarkerName,
                               _Output_GetMarkerGlobalTranslation.Translation[ 0 ],
                               _Output_GetMarkerGlobalTranslation.Translation[ 1 ],
                               _Output_GetMarkerGlobalTranslation.Translation[ 2 ],
                               _Output_GetMarkerGlobalTranslation.Occluded );

          if( bReadRayData )
          {
            Output_GetMarkerRayContributionCount _Output_GetMarkerRayContributionCount
              = MyClient.GetMarkerRayContributionCount(SubjectName, MarkerName);
            if( _Output_GetMarkerRayContributionCount.Result == Result.Success )
            {
              String ContributionsOutput = "      Contributed to by: ";
              for (uint ContributionIndex = 0; ContributionIndex < _Output_GetMarkerRayContributionCount.RayContributionsCount; ++ContributionIndex)
              {
                Output_GetMarkerRayContribution _Output_GetMarkerRayContribution =
                  MyClient.GetMarkerRayContribution(SubjectName, MarkerName, ContributionIndex);
                ContributionsOutput += "ID:" + _Output_GetMarkerRayContribution.CameraID + " Index:" + _Output_GetMarkerRayContribution.CentroidIndex + " ";
              }
              Console.WriteLine( ContributionsOutput );
            }

          }

          }
        }

        // Get the unlabeled markers
        uint UnlabeledMarkerCount = MyClient.GetUnlabeledMarkerCount().MarkerCount;
        Console.WriteLine( "  Unlabeled Markers ({0}):", UnlabeledMarkerCount );
        for( uint UnlabeledMarkerIndex = 0 ; UnlabeledMarkerIndex < UnlabeledMarkerCount ; ++UnlabeledMarkerIndex )
        {
          // Get the global marker translation
          Output_GetUnlabeledMarkerGlobalTranslation _Output_GetUnlabeledMarkerGlobalTranslation 
            = MyClient.GetUnlabeledMarkerGlobalTranslation( UnlabeledMarkerIndex );

          if (bMarkerTrajIds)
          {
            Console.WriteLine("    Marker #{0}: ({1}, {2}, {3}): Traj ID {4}",
                               UnlabeledMarkerIndex,
                               _Output_GetUnlabeledMarkerGlobalTranslation.Translation[0],
                               _Output_GetUnlabeledMarkerGlobalTranslation.Translation[1],
                               _Output_GetUnlabeledMarkerGlobalTranslation.Translation[2],
                               _Output_GetUnlabeledMarkerGlobalTranslation.MarkerID );

          }
          else
          {
            Console.WriteLine("    Marker #{0}: ({1}, {2}, {3})",
                               UnlabeledMarkerIndex,
                               _Output_GetUnlabeledMarkerGlobalTranslation.Translation[0],
                               _Output_GetUnlabeledMarkerGlobalTranslation.Translation[1],
                               _Output_GetUnlabeledMarkerGlobalTranslation.Translation[2]);
          }
        }

        // Get the Labeled markers
        uint LabeledMarkerCount = MyClient.GetLabeledMarkerCount().MarkerCount;
        Console.WriteLine("  Labeled Markers ({0}):", LabeledMarkerCount);
        for (uint LabeledMarkerIndex = 0; LabeledMarkerIndex < LabeledMarkerCount; ++LabeledMarkerIndex)
        {
          // Get the global marker translation
          Output_GetLabeledMarkerGlobalTranslation _Output_GetLabeledMarkerGlobalTranslation
            = MyClient.GetLabeledMarkerGlobalTranslation(LabeledMarkerIndex);

          if (bMarkerTrajIds)
          {
            Console.WriteLine("    Marker #{0}: ({1}, {2}, {3}): Traj ID {4}",
                               LabeledMarkerIndex,
                               _Output_GetLabeledMarkerGlobalTranslation.Translation[0],
                               _Output_GetLabeledMarkerGlobalTranslation.Translation[1],
                               _Output_GetLabeledMarkerGlobalTranslation.Translation[2],
                               _Output_GetLabeledMarkerGlobalTranslation.MarkerID );

          }
          else
          {
            Console.WriteLine("    Marker #{0}: ({1}, {2}, {3})",
                               LabeledMarkerIndex,
                               _Output_GetLabeledMarkerGlobalTranslation.Translation[0],
                               _Output_GetLabeledMarkerGlobalTranslation.Translation[1],
                               _Output_GetLabeledMarkerGlobalTranslation.Translation[2]);
          }
        }

        // Count the number of devices
        uint DeviceCount = MyClient.GetDeviceCount().DeviceCount;
        Console.WriteLine("  Devices ({0}):", DeviceCount);
        for( uint DeviceIndex = 0 ; DeviceIndex < DeviceCount ; ++DeviceIndex )
        {
          Console.WriteLine( "    Device #{0}:", DeviceIndex );

          // Get the device name and type
          Output_GetDeviceName _Output_GetDeviceName = MyClient.GetDeviceName( DeviceIndex );
          Console.WriteLine( "      Name: {0}", _Output_GetDeviceName.DeviceName );
          Console.WriteLine( "      Type: {0}", Adapt( _Output_GetDeviceName.DeviceType ) );

          // Count the number of device outputs
          uint DeviceOutputCount = MyClient.GetDeviceOutputCount( _Output_GetDeviceName.DeviceName ).DeviceOutputCount;
          Console.WriteLine( "      Device Outputs ({0}):", DeviceOutputCount );
          for( uint DeviceOutputIndex = 0 ; DeviceOutputIndex < DeviceOutputCount ; ++DeviceOutputIndex )
          {
            // Get the device output name and unit
            Output_GetDeviceOutputComponentName _Output_GetDeviceOutputComponentName = 
              MyClient.GetDeviceOutputComponentName( _Output_GetDeviceName.DeviceName, DeviceOutputIndex );

            // Get the number of subsamples for this output.
            uint DeviceOutputSubsamples = 
              MyClient.GetDeviceOutputSubsamples( _Output_GetDeviceName.DeviceName,
                                                  _Output_GetDeviceOutputComponentName.DeviceOutputName,
                                                  _Output_GetDeviceOutputComponentName.DeviceOutputComponentName ).DeviceOutputSubsamples;
            
            Console.WriteLine("      Device Output #{0}:", DeviceOutputIndex);
            
            Console.WriteLine("      Samples ({0}):", DeviceOutputSubsamples);
            
            // Display all the subsamples.
            for (uint DeviceOutputSubsample = 0; DeviceOutputSubsample < DeviceOutputSubsamples; ++DeviceOutputSubsample)
            {
                Console.WriteLine("        Sample #{0}:", DeviceOutputSubsample);

                // Get the device output value
                Output_GetDeviceOutputValue _Output_GetDeviceOutputValue =
                  MyClient.GetDeviceOutputValue( _Output_GetDeviceName.DeviceName,
                                                 _Output_GetDeviceOutputComponentName.DeviceOutputName,
                                                 _Output_GetDeviceOutputComponentName.DeviceOutputComponentName, 
                                                 DeviceOutputSubsample );

                Console.WriteLine("          '{0}' {1} {2} {3}",
                                   _Output_GetDeviceOutputComponentName.DeviceOutputComponentName,
                                   _Output_GetDeviceOutputValue.Value,
                                   Adapt(_Output_GetDeviceOutputComponentName.DeviceOutputUnit),
                                   _Output_GetDeviceOutputValue.Occluded);
            }
          }
        }
  

        // Count the number of force plates
        uint ForcePlateCount = MyClient.GetForcePlateCount().ForcePlateCount;
        Console.WriteLine("  Force Plates: ({0})", ForcePlateCount);
        for (uint ForcePlateIndex = 0; ForcePlateIndex < ForcePlateCount; ++ForcePlateIndex)
        {
          Console.WriteLine("    Force Plate #{0}:", ForcePlateIndex);

          uint ForcePlateSubsamples = MyClient.GetForcePlateSubsamples(ForcePlateIndex).ForcePlateSubsamples;

          Console.WriteLine("    Samples ({0}):", ForcePlateSubsamples);

          // Display all the subsamples for the plate.
          for (uint ForcePlateSubsample = 0; ForcePlateSubsample < ForcePlateSubsamples; ++ForcePlateSubsample)
          {
              Console.WriteLine("      Sample #{0}:", ForcePlateSubsample);

              // Get the forces, moments and centre of pressure.
              // These are output in global coordinates.
              Output_GetGlobalForceVector _Output_GetGlobalForceVector = MyClient.GetGlobalForceVector(ForcePlateIndex, ForcePlateSubsample);
              Console.WriteLine("        Force ({0}, {1}, {2})",
                                 _Output_GetGlobalForceVector.ForceVector[0],
                                 _Output_GetGlobalForceVector.ForceVector[1],
                                 _Output_GetGlobalForceVector.ForceVector[2]);

              Output_GetGlobalMomentVector _Output_GetGlobalMomentVector = MyClient.GetGlobalMomentVector(ForcePlateIndex, ForcePlateSubsample);
              Console.WriteLine("        Moment ({0}, {1}, {2})",
                                 _Output_GetGlobalMomentVector.MomentVector[0],
                                 _Output_GetGlobalMomentVector.MomentVector[1],
                                 _Output_GetGlobalMomentVector.MomentVector[2]);

              Output_GetGlobalCentreOfPressure _Output_GetGlobalCentreOfPressure = MyClient.GetGlobalCentreOfPressure(ForcePlateIndex, ForcePlateSubsample);
              Console.WriteLine("        CoP ({0}, {1}, {2})",
                                 _Output_GetGlobalCentreOfPressure.CentreOfPressure[0],
                                 _Output_GetGlobalCentreOfPressure.CentreOfPressure[1],
                                 _Output_GetGlobalCentreOfPressure.CentreOfPressure[2]);
          }
        }

        // Count the number of eye trackers
        uint EyeTrackerCount = MyClient.GetEyeTrackerCount().EyeTrackerCount;
        Console.WriteLine("  Eye Trackers: ({0})", EyeTrackerCount);
        for (uint EyeTrackerIndex = 0; EyeTrackerIndex < EyeTrackerCount; ++EyeTrackerIndex)
        {
          Console.WriteLine("    Eye Tracker #{0}:", EyeTrackerIndex);

          // Get the eye position and gaze direction.
          // These are output in global coordinates.
          Output_GetEyeTrackerGlobalPosition _Output_GetEyeTrackerGlobalPosition = MyClient.GetEyeTrackerGlobalPosition(EyeTrackerIndex);
          Console.WriteLine("      Position ({0}, {1}, {2}) {3}",
                             _Output_GetEyeTrackerGlobalPosition.Position[0],
                             _Output_GetEyeTrackerGlobalPosition.Position[1],
                             _Output_GetEyeTrackerGlobalPosition.Position[2],
                             _Output_GetEyeTrackerGlobalPosition.Occluded );

          Output_GetEyeTrackerGlobalGazeVector _Output_GetEyeTrackerGlobalGazeVector = MyClient.GetEyeTrackerGlobalGazeVector(EyeTrackerIndex);
          Console.WriteLine("      Gaze ({0}, {1}, {2}) {3}",
                             _Output_GetEyeTrackerGlobalGazeVector.GazeVector[0],
                             _Output_GetEyeTrackerGlobalGazeVector.GazeVector[1],
                             _Output_GetEyeTrackerGlobalGazeVector.GazeVector[2],
                             _Output_GetEyeTrackerGlobalGazeVector.Occluded );
        }

        if( bReadCentroids )
        {
          uint CameraCount = MyClient.GetCameraCount().CameraCount;
          Console.WriteLine("Cameras({0}):", CameraCount);

          for (uint CameraIndex = 0; CameraIndex < CameraCount; ++CameraIndex)
          {
            Console.WriteLine("  Camera #{0}:", CameraIndex);

            string CameraName = MyClient.GetCameraName(CameraIndex).CameraName;
            Console.WriteLine("    Name: {0}", CameraName);

            Console.WriteLine("    Id: {0}", MyClient.GetCameraId(CameraName).CameraId );
            Console.WriteLine("    User Id: {0}", MyClient.GetCameraUserId(CameraName).CameraUserId );
            Console.WriteLine("    Type: {0}", MyClient.GetCameraType(CameraName).CameraType );
            Console.WriteLine("    Display Name: {0}", MyClient.GetCameraDisplayName(CameraName).CameraDisplayName );
            Output_GetCameraResolution _Output_GetCameraResolution = MyClient.GetCameraResolution(CameraName);
            Console.WriteLine("    Resolution: {0} x {1}", _Output_GetCameraResolution.ResolutionX, _Output_GetCameraResolution.ResolutionY );
            Console.WriteLine("    Is Video Camera: {0}", (MyClient.GetIsVideoCamera(CameraName).IsVideoCamera ? "true" : "false") );

            uint CentroidCount = MyClient.GetCentroidCount(CameraName).CentroidCount;
            Console.WriteLine("    Centroids({0}):", CentroidCount);

            for (uint CentroidIndex = 0; CentroidIndex < CentroidCount; ++CentroidIndex)
            {
              Console.WriteLine("      Centroid #{0}:", CentroidIndex);

              Output_GetCentroidPosition _Output_GetCentroidPosition = MyClient.GetCentroidPosition(CameraName, CentroidIndex);
              Console.WriteLine("        Position: ({0}, {1})", _Output_GetCentroidPosition.CentroidPosition[0], _Output_GetCentroidPosition.CentroidPosition[1]);
              Console.WriteLine("        Radius: ({0})", _Output_GetCentroidPosition.Radius);
              //Console.WriteLine( "        Accuracy: ({0})", _Output_GetCentroidPosition.Accuracy );
              Output_GetCentroidWeight _Output_GetCentroidWeight = MyClient.GetCentroidWeight(CameraName, CentroidIndex);
              if( _Output_GetCentroidWeight.Result == Result.Success )
              {
                Console.WriteLine("        Weighting: {0}", _Output_GetCentroidWeight.Weight);
              }
            }

            if (bReadGreyscaleData)
            {
              uint BlobCount = MyClient.GetGreyscaleBlobCount(CameraName).BlobCount;
              Console.WriteLine("    Blobs({0}", BlobCount );

              Output_GetCameraSensorMode _Output_SensorMode = MyClient.GetCameraSensorMode(CameraName);
              if (_Output_SensorMode.Result == Result.Success)
              {
                Console.WriteLine( "   SensorMode( {0} ", _Output_SensorMode.SensorMode );
              }

              Output_GetCameraWindowSize _Output_CameraWindowSize = MyClient.GetCameraWindowSize(CameraName);
              if (_Output_CameraWindowSize.Result == Result.Success)
              {
                Console.WriteLine( "   WindowSize( {0}, {1}, {2}, {3} ):",
                    _Output_CameraWindowSize.WindowStartX,
                    _Output_CameraWindowSize.WindowStartY,
                    _Output_CameraWindowSize.WindowWidth,
                    _Output_CameraWindowSize.WindowHeight );
              }

              Output_GetGreyscaleBlobSubsampleInfo _Output_SubsampleInfo = MyClient.GetGreyscaleBlobSubsampleInfo(CameraName);
              if (_Output_SubsampleInfo.Result == Result.Success)
              {
                    Console.WriteLine("   WindowSize( {0}, {1}, {2}, {3} ):",
                        _Output_SubsampleInfo.TwiceOffsetX,
                        _Output_SubsampleInfo.TwiceOffsetX,
                        _Output_SubsampleInfo.SensorPixelsPerImagePixelX,
                        _Output_SubsampleInfo.SensorPixelsPerImagePixelY );
              }

              for (uint BlobIndex = 0; BlobIndex < BlobCount; ++BlobIndex)
              {
                Console.WriteLine("      Blob #{0}:", BlobIndex );

                Output_GetGreyscaleBlob _Output_GetGreyscaleBlob = MyClient.GetGreyscaleBlob(CameraName, BlobIndex);
                if (_Output_GetGreyscaleBlob.Result == Result.Success)
                {
                  // Don't print out the whole blob
                  Console.WriteLine("        # Line Positions X: {0}", _Output_GetGreyscaleBlob.BlobLinePositionsX.Count);
                  Console.WriteLine("        # Line Positions Y: {0}",  _Output_GetGreyscaleBlob.BlobLinePositionsY.Count);
                  Console.WriteLine("        # Pixel Values: {0}", _Output_GetGreyscaleBlob.BlobLinePixelValues.Count);
                }
              }
            }

            if (bReadVideoData)
            {
              Output_GetVideoFrame _Output_GetVideoFrame = MyClient.GetVideoFrame(CameraName);
              if (_Output_GetVideoFrame.Result == Result.Success)
              {
                Console.WriteLine("    Video Frame:");
                Console.WriteLine("      Width: {0}", _Output_GetVideoFrame.m_Width);
                Console.WriteLine("      Height: {0}", _Output_GetVideoFrame.m_Height);
                Console.WriteLine("      Frame Id: {0}",  _Output_GetVideoFrame.m_FrameID);
                Console.WriteLine("      Format: {0}", _Output_GetVideoFrame.m_Format);
                Console.WriteLine("      Data: {0} samples", _Output_GetVideoFrame.m_Data.Count);
              }
            }

          }
        }
      }

      if( TransmitMulticast )
      {
        MyClient.StopTransmittingMulticast();
      }

      MyClient.DisableSegmentData();
      MyClient.DisableMarkerData();
      MyClient.DisableUnlabeledMarkerData();
      MyClient.DisableDeviceData();
      if (bReadCentroids)
      {
        MyClient.DisableCentroidData();
      }
      if (bReadRayData)
      {
        MyClient.DisableMarkerRayData();
      }
      if (bReadGreyscaleData)
      {
        MyClient.DisableGreyscaleData();
      }
      if (bReadVideoData)
      {
        MyClient.DisableVideoData();
      }

      // Disconnect and dispose
      MyClient.Disconnect();
      MyClient = null;
    }
  }
}

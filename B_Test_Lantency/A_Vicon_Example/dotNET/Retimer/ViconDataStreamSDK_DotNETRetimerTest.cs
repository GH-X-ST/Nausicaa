///////////////////////////////////////////////////////////////////////////////
//
// Copyright (C) OMG Plc 2017.
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
using System.Threading;
using ViconDataStreamSDK.DotNET;

namespace CSharpRetimingClient
{
class Program
{

  static void PrintSubjects( ViconDataStreamSDK.DotNET.RetimingClient MyClient )
  {
    // Count the number of subjects
    uint SubjectCount = MyClient.GetSubjectCount().SubjectCount;
    Console.WriteLine( "Subjects ({0}):", SubjectCount );
    for (uint SubjectIndex = 0; SubjectIndex < SubjectCount; ++SubjectIndex)
    {
      Console.WriteLine("  Subject #{0}", SubjectIndex);

      // Get the subject name
      string SubjectName = MyClient.GetSubjectName(SubjectIndex).SubjectName;
      Console.WriteLine("    Name: {0}", SubjectName);

      // Get the root segment
      string RootSegment = MyClient.GetSubjectRootSegmentName(SubjectName).SegmentName;
      Console.WriteLine("    Root Segment: {0}", RootSegment);

      // Count the number of segments
      uint SegmentCount = MyClient.GetSegmentCount(SubjectName).SegmentCount;
      Console.WriteLine("    Segments ({0}):", SegmentCount);
      for (uint SegmentIndex = 0; SegmentIndex < SegmentCount; ++SegmentIndex)
      {
        Console.WriteLine("      Segment #{0}", SegmentIndex);

        // Get the segment name
        string SegmentName = MyClient.GetSegmentName(SubjectName, SegmentIndex).SegmentName;
        Console.WriteLine("        Name: {0}", SegmentName);

        // Get the segment parent
        string SegmentParentName = MyClient.GetSegmentParentName(SubjectName, SegmentName).SegmentName;
        Console.WriteLine("        Parent: {0}", SegmentParentName);

        // Get the segment's children
        uint ChildCount = MyClient.GetSegmentChildCount(SubjectName, SegmentName).SegmentCount;
        Console.WriteLine("     Children ({0}):", ChildCount);
        for (uint ChildIndex = 0; ChildIndex < ChildCount; ++ChildIndex)
        {
          string ChildName = MyClient.GetSegmentChildName(SubjectName, SegmentName, ChildIndex).SegmentName;
          Console.WriteLine("       {0}", ChildName);
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
          MyClient.GetSegmentGlobalTranslation(SubjectName, SegmentName);
        Console.WriteLine("        Global Translation: ({0},{1},{2}) {3}",
                            _Output_GetSegmentGlobalTranslation.Translation[0],
                            _Output_GetSegmentGlobalTranslation.Translation[1],
                            _Output_GetSegmentGlobalTranslation.Translation[2],
                            _Output_GetSegmentGlobalTranslation.Occluded);

        // Get the global segment rotation in helical co-ordinates
        Output_GetSegmentGlobalRotationHelical _Output_GetSegmentGlobalRotationHelical =
          MyClient.GetSegmentGlobalRotationHelical(SubjectName, SegmentName);
        Console.WriteLine("        Global Rotation Helical: ({0},{1},{2}) {3}",
                            _Output_GetSegmentGlobalRotationHelical.Rotation[0],
                            _Output_GetSegmentGlobalRotationHelical.Rotation[1],
                            _Output_GetSegmentGlobalRotationHelical.Rotation[2],
                            _Output_GetSegmentGlobalRotationHelical.Occluded);

        // Get the global segment rotation as a matrix
        Output_GetSegmentGlobalRotationMatrix _Output_GetSegmentGlobalRotationMatrix =
          MyClient.GetSegmentGlobalRotationMatrix(SubjectName, SegmentName);
        Console.WriteLine("        Global Rotation Matrix: ({0},{1},{2},{3},{4},{5},{6},{7},{8}) {9}",
                            _Output_GetSegmentGlobalRotationMatrix.Rotation[0],
                            _Output_GetSegmentGlobalRotationMatrix.Rotation[1],
                            _Output_GetSegmentGlobalRotationMatrix.Rotation[2],
                            _Output_GetSegmentGlobalRotationMatrix.Rotation[3],
                            _Output_GetSegmentGlobalRotationMatrix.Rotation[4],
                            _Output_GetSegmentGlobalRotationMatrix.Rotation[5],
                            _Output_GetSegmentGlobalRotationMatrix.Rotation[6],
                            _Output_GetSegmentGlobalRotationMatrix.Rotation[7],
                            _Output_GetSegmentGlobalRotationMatrix.Rotation[8],
                            _Output_GetSegmentGlobalRotationMatrix.Occluded);

        // Get the global segment rotation in quaternion co-ordinates
        Output_GetSegmentGlobalRotationQuaternion _Output_GetSegmentGlobalRotationQuaternion =
          MyClient.GetSegmentGlobalRotationQuaternion(SubjectName, SegmentName);
        Console.WriteLine("        Global Rotation Quaternion: ({0},{1},{2},{3}) {4}",
                            _Output_GetSegmentGlobalRotationQuaternion.Rotation[0],
                            _Output_GetSegmentGlobalRotationQuaternion.Rotation[1],
                            _Output_GetSegmentGlobalRotationQuaternion.Rotation[2],
                            _Output_GetSegmentGlobalRotationQuaternion.Rotation[3],
                            _Output_GetSegmentGlobalRotationQuaternion.Occluded);

        // Get the global segment rotation in EulerXYZ co-ordinates
        Output_GetSegmentGlobalRotationEulerXYZ _Output_GetSegmentGlobalRotationEulerXYZ =
          MyClient.GetSegmentGlobalRotationEulerXYZ(SubjectName, SegmentName);
        Console.WriteLine("        Global Rotation EulerXYZ: ({0},{1},{2}) {3}",
                            _Output_GetSegmentGlobalRotationEulerXYZ.Rotation[0],
                            _Output_GetSegmentGlobalRotationEulerXYZ.Rotation[1],
                            _Output_GetSegmentGlobalRotationEulerXYZ.Rotation[2],
                            _Output_GetSegmentGlobalRotationEulerXYZ.Occluded);

        // Get the local segment translation
        Output_GetSegmentLocalTranslation _Output_GetSegmentLocalTranslation =
          MyClient.GetSegmentLocalTranslation(SubjectName, SegmentName);
        Console.WriteLine("        Local Translation: ({0},{1},{2}) {3}",
                            _Output_GetSegmentLocalTranslation.Translation[0],
                            _Output_GetSegmentLocalTranslation.Translation[1],
                            _Output_GetSegmentLocalTranslation.Translation[2],
                            _Output_GetSegmentLocalTranslation.Occluded);

        // Get the local segment rotation in helical co-ordinates
        Output_GetSegmentLocalRotationHelical _Output_GetSegmentLocalRotationHelical =
          MyClient.GetSegmentLocalRotationHelical(SubjectName, SegmentName);
        Console.WriteLine("        Local Rotation Helical: ({0},{1},{2}) {3}",
                            _Output_GetSegmentLocalRotationHelical.Rotation[0],
                            _Output_GetSegmentLocalRotationHelical.Rotation[1],
                            _Output_GetSegmentLocalRotationHelical.Rotation[2],
                            _Output_GetSegmentLocalRotationHelical.Occluded);

        // Get the local segment rotation as a matrix
        Output_GetSegmentLocalRotationMatrix _Output_GetSegmentLocalRotationMatrix =
          MyClient.GetSegmentLocalRotationMatrix(SubjectName, SegmentName);
        Console.WriteLine("        Local Rotation Matrix: ({0},{1},{2},{3},{4},{5},{6},{7},{8}) {9}",
                            _Output_GetSegmentLocalRotationMatrix.Rotation[0],
                            _Output_GetSegmentLocalRotationMatrix.Rotation[1],
                            _Output_GetSegmentLocalRotationMatrix.Rotation[2],
                            _Output_GetSegmentLocalRotationMatrix.Rotation[3],
                            _Output_GetSegmentLocalRotationMatrix.Rotation[4],
                            _Output_GetSegmentLocalRotationMatrix.Rotation[5],
                            _Output_GetSegmentLocalRotationMatrix.Rotation[6],
                            _Output_GetSegmentLocalRotationMatrix.Rotation[7],
                            _Output_GetSegmentLocalRotationMatrix.Rotation[8],
                            _Output_GetSegmentLocalRotationMatrix.Occluded);

        // Get the local segment rotation in quaternion co-ordinates
        Output_GetSegmentLocalRotationQuaternion _Output_GetSegmentLocalRotationQuaternion =
          MyClient.GetSegmentLocalRotationQuaternion(SubjectName, SegmentName);
        Console.WriteLine("        Local Rotation Quaternion: ({0},{1},{2},{3}) {4}",
                            _Output_GetSegmentLocalRotationQuaternion.Rotation[0],
                            _Output_GetSegmentLocalRotationQuaternion.Rotation[1],
                            _Output_GetSegmentLocalRotationQuaternion.Rotation[2],
                            _Output_GetSegmentLocalRotationQuaternion.Rotation[3],
                            _Output_GetSegmentLocalRotationQuaternion.Occluded);

        // Get the local segment rotation in EulerXYZ co-ordinates
        Output_GetSegmentLocalRotationEulerXYZ _Output_GetSegmentLocalRotationEulerXYZ =
          MyClient.GetSegmentLocalRotationEulerXYZ(SubjectName, SegmentName);
        Console.WriteLine("        Local Rotation EulerXYZ: ({0},{1},{2}) {3}",
                            _Output_GetSegmentLocalRotationEulerXYZ.Rotation[0],
                            _Output_GetSegmentLocalRotationEulerXYZ.Rotation[1],
                            _Output_GetSegmentLocalRotationEulerXYZ.Rotation[2],
                            _Output_GetSegmentLocalRotationEulerXYZ.Occluded);
      }
    }
  }

  static void Main( string[] args )
  {

    Console.WriteLine( "DSSDK DotNET API Retiming Test");

    String HostName = "localhost:801";
    if( args.Length > 0 )
    {
      HostName = args[0];
    }

    double FrameRate = -1;
    bool bLightweightSegments = false;
    List<String> FilteredSubjects = new List<string>();
    bool bSubjectFilterApplied = false;

      // parsing all the haptic arguments
      for ( int i = 1; i < args.Length; ++i )
    {
      if ( String.Compare( args[i], "--help" ) == 0 )
      {
        Console.WriteLine( "ViconDataStreamSDK_DotNETRetimerTest <HostName>: allowed options include:\n --framerate <frame rate>");
        return;
      }
    else if( ( String.Compare( args[i], "--framerate" ) == 0 ) )
    {
      ++i;
      if (i < args.Length)
      {
        try
        {
          FrameRate = Convert.ToDouble(args[i]);
        }
        catch (FormatException)
        {
          Console.WriteLine("Invalid argument for --framerate");
        }
        catch (OverflowException)
        {
          Console.WriteLine("Invalid argument for --framerate");
        }
      }
    }
    else if( ( String.Compare( args[i], "--lightweight") == 0 ) )
    {
      bLightweightSegments = true;
    }
    else if (String.Compare(args[i], "--subjects") == 0)
    {
      ++i;
      while (i < args.Length && String.Compare(args[i], 0, "--", 0, 2) != 0)
      {
        FilteredSubjects.Add(args[i]);
        ++i;
      }
    }
  }

    ViconDataStreamSDK.DotNET.RetimingClient Retimer = new ViconDataStreamSDK.DotNET.RetimingClient();

    if( bLightweightSegments )
    {
      Retimer.EnableLightweightSegmentData();
    }
    if (Retimer.Connect(HostName, FrameRate ).Result != Result.Success)
    {
      Console.WriteLine( "Could not connect to {0}", HostName );
      return;
    }


      bool bStop = false;

//      // The retimer may sometimes take a few frames to start producing data as it 
//      // needs to fill a buffer before it can produce output.
//      // We will use this flag to inhibit output during this spool up phase.
      bool bDataReceived = false;

      while( !bStop )
      {

        // We are running on a loop timer - use this method if you do not have an existing timer in your project
        if( FrameRate > 0 )
        {
          // Wait for a frame. This method blocks until the next frame period has elapsed.
          Output_WaitForFrame _Output_WaitForFrame = Retimer.WaitForFrame();
          if (!bDataReceived && _Output_WaitForFrame.Result == Result.Success)
          {
            bDataReceived = true;
            if (!bSubjectFilterApplied)
            {
              Retimer.ClearSubjectFilter();
              foreach (String Subject in FilteredSubjects)
              {
                Output_AddToSubjectFilter Result = Retimer.AddToSubjectFilter(Subject);
                bSubjectFilterApplied = bSubjectFilterApplied || Result.Result == ViconDataStreamSDK.DotNET.Result.Success;
              }
            }
          }
        }
        else
        {
          // This code path is for projects that already have some external signal driving the frame update; it may be an internal timer, or a display refresh call.
          Output_UpdateFrame _Output_UpdateFrame = Retimer.UpdateFrame();
          if (!bDataReceived && _Output_UpdateFrame.Result == Result.Success)
          {
            bDataReceived = true;
            if (!bSubjectFilterApplied)
            {
              Retimer.ClearSubjectFilter();
              foreach (String Subject in FilteredSubjects)
              {
                Output_AddToSubjectFilter Result = Retimer.AddToSubjectFilter(Subject);
                bSubjectFilterApplied = bSubjectFilterApplied || Result.Result == ViconDataStreamSDK.DotNET.Result.Success;
              }
            }
          }
          
          // Impose our own simple frame rate throttle, so that we do not overload the process
          Thread.Sleep( 10 );
        }

        if (bDataReceived)
        {
          PrintSubjects( Retimer );
        }
      }

    Retimer.Disconnect();

  }
}
}

/*/////////////////////////////////////////////////////////////////////////////
//
// Copyright (C) OMG Plc 2009.
// All rights reserved.  This software is protected by copyright
// law and international treaties.  No part of this software / document
// may be reproduced or distributed in any form or by any means,
// whether transiently or incidentally to some other use of this software,
// without the written permission of the copyright owner.
//
//////////////////////////////////////////////////////////////////////////////*/

#include "CClient.h"
#include "CRetimingClient.h"

#pragma warning( push )
#pragma warning( disable:4255 )
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <conio.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

#pragma warning( pop )

CString Adapt(const CBool i_Value)
{
  return i_Value ? "True" : "False";
}

void PrintSubjects(CRetimingClient * pClient)
{
  // Count the number of subjects
  COutput_GetSubjectCount SubjectCount;
  unsigned int SubjectIndex;

  RetimingClient_GetSubjectCount(pClient, &SubjectCount);
  printf("Subjects (%d):\n", SubjectCount.SubjectCount);
  for (SubjectIndex = 0; SubjectIndex < SubjectCount.SubjectCount; ++SubjectIndex)
  {
    printf("  Subject #%d\n", SubjectIndex);

    // Get the subject name
    char SubjectName[128];
    RetimingClient_GetSubjectName(pClient, SubjectIndex, 128, SubjectName);
    printf("    Name: %s\n", SubjectName);

    // Get the root segment
    char RootSegment[128];
    RetimingClient_GetSubjectRootSegmentName(pClient, SubjectName, 128, RootSegment);
    printf("    Root Segment: %s\n", RootSegment);

    // Count the number of segments
    COutput_GetSegmentCount SegmentCount;
    RetimingClient_GetSegmentCount(pClient, SubjectName, &SegmentCount);

    unsigned int SegmentIndex;
    printf("    Segments (%d):\n", SegmentCount.SegmentCount);
    for (SegmentIndex = 0; SegmentIndex < SegmentCount.SegmentCount; ++SegmentIndex)
    {
      printf("      Segment #%d\n", SegmentIndex);

      // Get the segment name
      char SegmentName[128];
      RetimingClient_GetSegmentName(pClient, SubjectName, SegmentIndex, 128, SegmentName);
      printf("        Name: %s\n", SegmentName);

      // Get the segment parent
      char SegmentParentName[128];
      RetimingClient_GetSegmentParentName(pClient, SubjectName, SegmentName, 128, SegmentParentName);
      printf("        Parent: %s\n", SegmentParentName);

      // Get the segment's children
      COutput_GetSegmentChildCount ChildCount;
      RetimingClient_GetSegmentChildCount(pClient, SubjectName, SegmentName, &ChildCount);
      unsigned int ChildIndex;
      printf("     Children (%d):\n", ChildCount.SegmentCount);
      for (ChildIndex = 0; ChildIndex < ChildCount.SegmentCount; ++ChildIndex)
      {
        char SegmentChildName[128];
        RetimingClient_GetSegmentChildName(pClient, SubjectName, SegmentName, ChildIndex, 128, SegmentChildName);
        printf("       %s\n", SegmentChildName);
      }

      // Get the static segment translation
      COutput_GetSegmentStaticTranslation _Output_GetSegmentStaticTranslation;
      RetimingClient_GetSegmentStaticTranslation(pClient, SubjectName, SegmentName, &_Output_GetSegmentStaticTranslation);
      printf("        Static Translation: (%f, %f, %f)\n"
        , _Output_GetSegmentStaticTranslation.Translation[0]
        , _Output_GetSegmentStaticTranslation.Translation[1]
        , _Output_GetSegmentStaticTranslation.Translation[2]
        );

      // Get the static segment rotation in helical co-ordinates
      COutput_GetSegmentStaticRotationHelical _Output_GetSegmentStaticRotationHelical;
      RetimingClient_GetSegmentStaticRotationHelical(pClient, SubjectName, SegmentName, &_Output_GetSegmentStaticRotationHelical);
      printf("        Static Rotation Helical: (%f, %f, %f)\n"
        , _Output_GetSegmentStaticRotationHelical.Rotation[0]
        , _Output_GetSegmentStaticRotationHelical.Rotation[1]
        , _Output_GetSegmentStaticRotationHelical.Rotation[2]
        );

      // Get the static segment rotation as a matrix
      COutput_GetSegmentStaticRotationMatrix _Output_GetSegmentStaticRotationMatrix;
      RetimingClient_GetSegmentStaticRotationMatrix(pClient, SubjectName, SegmentName, &_Output_GetSegmentStaticRotationMatrix);
      printf("        Static Rotation Matrix: (%f, %f, %f, %f, %f, %f, %f, %f, %f)\n"
        , _Output_GetSegmentStaticRotationMatrix.Rotation[0]
        , _Output_GetSegmentStaticRotationMatrix.Rotation[1]
        , _Output_GetSegmentStaticRotationMatrix.Rotation[2]
        , _Output_GetSegmentStaticRotationMatrix.Rotation[3]
        , _Output_GetSegmentStaticRotationMatrix.Rotation[4]
        , _Output_GetSegmentStaticRotationMatrix.Rotation[5]
        , _Output_GetSegmentStaticRotationMatrix.Rotation[6]
        , _Output_GetSegmentStaticRotationMatrix.Rotation[7]
        , _Output_GetSegmentStaticRotationMatrix.Rotation[8]
        );

      // Get the static segment rotation in quaternion co-ordinates
      COutput_GetSegmentStaticRotationQuaternion _Output_GetSegmentStaticRotationQuaternion;
      RetimingClient_GetSegmentStaticRotationQuaternion(pClient, SubjectName, SegmentName, &_Output_GetSegmentStaticRotationQuaternion);
      printf("        Static Rotation Quaternion: (%f, %f, %f, %f)\n"
        , _Output_GetSegmentStaticRotationQuaternion.Rotation[0]
        , _Output_GetSegmentStaticRotationQuaternion.Rotation[1]
        , _Output_GetSegmentStaticRotationQuaternion.Rotation[2]
        , _Output_GetSegmentStaticRotationQuaternion.Rotation[3]
        );

      // Get the static segment rotation in EulerXYZ co-ordinates
      COutput_GetSegmentStaticRotationEulerXYZ _Output_GetSegmentStaticRotationEulerXYZ;
      RetimingClient_GetSegmentStaticRotationEulerXYZ(pClient, SubjectName, SegmentName, &_Output_GetSegmentStaticRotationEulerXYZ);
      printf("        Static Rotation EulerXYZ: (%f, %f, %f)\n"
        , _Output_GetSegmentStaticRotationEulerXYZ.Rotation[0]
        , _Output_GetSegmentStaticRotationEulerXYZ.Rotation[1]
        , _Output_GetSegmentStaticRotationEulerXYZ.Rotation[2]
        );

      // Get the global segment translation
      COutput_GetSegmentGlobalTranslation _Output_GetSegmentGlobalTranslation;
      RetimingClient_GetSegmentGlobalTranslation(pClient, SubjectName, SegmentName, &_Output_GetSegmentGlobalTranslation);
      printf("        Global Translation: (%f, %f, %f) %s\n"
        , _Output_GetSegmentGlobalTranslation.Translation[0]
        , _Output_GetSegmentGlobalTranslation.Translation[1]
        , _Output_GetSegmentGlobalTranslation.Translation[2]
        , Adapt(_Output_GetSegmentGlobalTranslation.Occluded)
        );

      // Get the global segment rotation in helical co-ordinates
      COutput_GetSegmentGlobalRotationHelical _Output_GetSegmentGlobalRotationHelical;
      RetimingClient_GetSegmentGlobalRotationHelical(pClient, SubjectName, SegmentName, &_Output_GetSegmentGlobalRotationHelical);
      printf("        Global Rotation Helical: (%f, %f, %f) %s\n"
        , _Output_GetSegmentGlobalRotationHelical.Rotation[0]
        , _Output_GetSegmentGlobalRotationHelical.Rotation[1]
        , _Output_GetSegmentGlobalRotationHelical.Rotation[2]
        , Adapt(_Output_GetSegmentGlobalRotationHelical.Occluded)
        );

      // Get the global segment rotation as a matrix
      COutput_GetSegmentGlobalRotationMatrix _Output_GetSegmentGlobalRotationMatrix;
      RetimingClient_GetSegmentGlobalRotationMatrix(pClient, SubjectName, SegmentName, &_Output_GetSegmentGlobalRotationMatrix);
      printf("        Global Rotation Matrix: (%f, %f, %f, %f, %f, %f, %f, %f, %f) %s\n"
        , _Output_GetSegmentGlobalRotationMatrix.Rotation[0]
        , _Output_GetSegmentGlobalRotationMatrix.Rotation[1]
        , _Output_GetSegmentGlobalRotationMatrix.Rotation[2]
        , _Output_GetSegmentGlobalRotationMatrix.Rotation[3]
        , _Output_GetSegmentGlobalRotationMatrix.Rotation[4]
        , _Output_GetSegmentGlobalRotationMatrix.Rotation[5]
        , _Output_GetSegmentGlobalRotationMatrix.Rotation[6]
        , _Output_GetSegmentGlobalRotationMatrix.Rotation[7]
        , _Output_GetSegmentGlobalRotationMatrix.Rotation[8]
        , Adapt(_Output_GetSegmentGlobalRotationMatrix.Occluded));

      // Get the global segment rotation in quaternion co-ordinates
      COutput_GetSegmentGlobalRotationQuaternion _Output_GetSegmentGlobalRotationQuaternion;
      RetimingClient_GetSegmentGlobalRotationQuaternion(pClient, SubjectName, SegmentName, &_Output_GetSegmentGlobalRotationQuaternion);
      printf("        Global Rotation Quaternion: (%f, %f, %f, %f) %s\n"
        , _Output_GetSegmentGlobalRotationQuaternion.Rotation[0]
        , _Output_GetSegmentGlobalRotationQuaternion.Rotation[1]
        , _Output_GetSegmentGlobalRotationQuaternion.Rotation[2]
        , _Output_GetSegmentGlobalRotationQuaternion.Rotation[3]
        , Adapt(_Output_GetSegmentGlobalRotationQuaternion.Occluded));

      // Get the global segment rotation in EulerXYZ co-ordinates
      COutput_GetSegmentGlobalRotationEulerXYZ _Output_GetSegmentGlobalRotationEulerXYZ;
      RetimingClient_GetSegmentGlobalRotationEulerXYZ(pClient, SubjectName, SegmentName, &_Output_GetSegmentGlobalRotationEulerXYZ);
      printf("        Global Rotation EulerXYZ: (%f, %f, %f) %s\n"
        , _Output_GetSegmentGlobalRotationEulerXYZ.Rotation[0]
        , _Output_GetSegmentGlobalRotationEulerXYZ.Rotation[1]
        , _Output_GetSegmentGlobalRotationEulerXYZ.Rotation[2]
        , Adapt(_Output_GetSegmentGlobalRotationEulerXYZ.Occluded));

      // Get the local segment translation
      COutput_GetSegmentLocalTranslation _Output_GetSegmentLocalTranslation;
      RetimingClient_GetSegmentLocalTranslation(pClient, SubjectName, SegmentName, &_Output_GetSegmentLocalTranslation);
      printf("        Local Translation: (%f, %f, %f) %s\n"
        , _Output_GetSegmentLocalTranslation.Translation[0]
        , _Output_GetSegmentLocalTranslation.Translation[1]
        , _Output_GetSegmentLocalTranslation.Translation[2]
        , Adapt(_Output_GetSegmentLocalTranslation.Occluded));

      // Get the local segment rotation in helical co-ordinates
      COutput_GetSegmentLocalRotationHelical _Output_GetSegmentLocalRotationHelical;
      RetimingClient_GetSegmentLocalRotationHelical(pClient, SubjectName, SegmentName, &_Output_GetSegmentLocalRotationHelical);
      printf("        Local Rotation Helical: (%f, %f, %f) %s\n"
        , _Output_GetSegmentLocalRotationHelical.Rotation[0]
        , _Output_GetSegmentLocalRotationHelical.Rotation[1]
        , _Output_GetSegmentLocalRotationHelical.Rotation[2]
        , Adapt(_Output_GetSegmentLocalRotationHelical.Occluded));

      // Get the local segment rotation as a matrix
      COutput_GetSegmentLocalRotationMatrix _Output_GetSegmentLocalRotationMatrix;
      RetimingClient_GetSegmentLocalRotationMatrix(pClient, SubjectName, SegmentName, &_Output_GetSegmentLocalRotationMatrix);
      printf("        Local Rotation Matrix: (%f, %f, %f, %f, %f, %f, %f, %f, %f) %s\n"
        , _Output_GetSegmentLocalRotationMatrix.Rotation[0]
        , _Output_GetSegmentLocalRotationMatrix.Rotation[1]
        , _Output_GetSegmentLocalRotationMatrix.Rotation[2]
        , _Output_GetSegmentLocalRotationMatrix.Rotation[3]
        , _Output_GetSegmentLocalRotationMatrix.Rotation[4]
        , _Output_GetSegmentLocalRotationMatrix.Rotation[5]
        , _Output_GetSegmentLocalRotationMatrix.Rotation[6]
        , _Output_GetSegmentLocalRotationMatrix.Rotation[7]
        , _Output_GetSegmentLocalRotationMatrix.Rotation[8]
        , Adapt(_Output_GetSegmentLocalRotationMatrix.Occluded));

      // Get the local segment rotation in quaternion co-ordinates
      COutput_GetSegmentLocalRotationQuaternion _Output_GetSegmentLocalRotationQuaternion;
      RetimingClient_GetSegmentLocalRotationQuaternion(pClient, SubjectName, SegmentName, &_Output_GetSegmentLocalRotationQuaternion);
      printf("        Local Rotation Quaternion: (%f, %f, %f, %f) %s\n"
        , _Output_GetSegmentLocalRotationQuaternion.Rotation[0]
        , _Output_GetSegmentLocalRotationQuaternion.Rotation[1]
        , _Output_GetSegmentLocalRotationQuaternion.Rotation[2]
        , _Output_GetSegmentLocalRotationQuaternion.Rotation[3]
        , Adapt(_Output_GetSegmentLocalRotationQuaternion.Occluded));

      // Get the local segment rotation in EulerXYZ co-ordinates
      COutput_GetSegmentLocalRotationEulerXYZ _Output_GetSegmentLocalRotationEulerXYZ;
      RetimingClient_GetSegmentLocalRotationEulerXYZ(pClient, SubjectName, SegmentName, &_Output_GetSegmentLocalRotationEulerXYZ);
      printf("        Local Rotation EulerXYZ: (%f, %f, %f) %s\n"
        , _Output_GetSegmentLocalRotationEulerXYZ.Rotation[0]
        , _Output_GetSegmentLocalRotationEulerXYZ.Rotation[1]
        , _Output_GetSegmentLocalRotationEulerXYZ.Rotation[2]
        , Adapt(_Output_GetSegmentLocalRotationEulerXYZ.Occluded));
    }
  }
}

int main( int argc, char* argv[] )
{
  printf( "DSSDK C API Retiming Test\n" );

  CString HostName = "localhost:801";
  CBool bSubjectFilterApplied = 0;

  char** FilteredSubjects = NULL;
  int SubjectFilterSize = 0;

  double FrameRate = -1;
  CBool bLightweightSegments = 0;

  if( argc > 1 )
  {
    HostName = argv[1];

  }
  int a;
  for (a = 2; a < argc; ++a)
  {
    if (strcmp(argv[a], "--help") == 0)
    {
      printf("%s <HostName>: allowed options include:\n --framerate <frame rate> --help\n", argv[0] );
      return 0;
    }
    else if (strcmp(argv[a], "--framerate") == 0)
    {
      if (a < argc)
      {
        FrameRate = atoi(argv[++a]);
      }
    }
    else if ( strcmp( argv[a], "--lightweight" ) == 0 )
    {
      bLightweightSegments = 1;
    }
    else if (strcmp( argv[a], "--subjects" ) == 0)
    {
      ++a;
      if( a < argc )
      {
        size_t MaxSubjectSize = argc - a;
        FilteredSubjects = malloc( (argc - a) * sizeof( char* ) );
        size_t j;
        for ( j = 0; j < MaxSubjectSize; ++j)
        {
          FilteredSubjects[j] = NULL;
        }
      }
      int i = 0;
      while (a < argc)
      {
        if (strncmp( argv[a], "--", 2 ) == 0)
        {
          --a;
          break;
        }
        // 
        char* Subject = argv[a];
        FilteredSubjects[i] = Subject;
        ++a;
        ++i;
      }
      SubjectFilterSize = i;
    }
  }

  CRetimingClient * pRetimingClient = pRetimingClient = RetimingClient_Create();

  if ( bLightweightSegments )
  {
    RetimingClient_EnableLightweightSegmentData( pRetimingClient );
  }

  CBool bConnected = 0;

  if( FrameRate > 0 )
  {
    bConnected = RetimingClient_ConnectAndStart(pRetimingClient, HostName, FrameRate) == Success ;
  }
  else
  {
    bConnected = RetimingClient_Connect(pRetimingClient, HostName) == Success;
  }

  if (!bConnected)
  {
    printf("Failed to connect\n");
    exit(1);
  }

  CBool bDataReceived = 0;

  while( bConnected )
  {
    if( FrameRate > 0 )
    {
      CEnum WaitResult = RetimingClient_WaitForFrame( pRetimingClient );
      if (!bDataReceived && (WaitResult == Success))
      {
        bDataReceived = 1;

        if (!bSubjectFilterApplied)
        {
          int subject_index;
          for ( subject_index = 0; subject_index < SubjectFilterSize; ++subject_index)
          {
            if (FilteredSubjects[subject_index] != NULL)
            {
              char * Subject = FilteredSubjects[subject_index];
              CEnum _Output_AddToSubjectFilter = Client_AddToSubjectFilter( pRetimingClient, Subject );
              bSubjectFilterApplied = bSubjectFilterApplied || _Output_AddToSubjectFilter == CSuccess;
            }
          }
        }
      }
    }
    else
    {
      CEnum UpdateResult = RetimingClient_UpdateFrame(pRetimingClient );
      if (!bDataReceived && (UpdateResult == Success))
      {
        bDataReceived = 1;

        if (!bSubjectFilterApplied)
        {
          int subject_index;
          for (subject_index = 0; subject_index < SubjectFilterSize; ++subject_index)
          {
            if (FilteredSubjects[subject_index] != NULL)
            {
              char * Subject = FilteredSubjects[subject_index];
              CEnum _Output_AddToSubjectFilter = Client_AddToSubjectFilter( pRetimingClient, Subject );
              bSubjectFilterApplied = bSubjectFilterApplied || _Output_AddToSubjectFilter == CSuccess;
            }
          }
        }
      }

      /* Sleep a little */
#ifdef _WIN32
      Sleep(10);
#else
      sleep(1);
#endif 
    }

    PrintSubjects(pRetimingClient);
  }


  RetimingClient_Disconnect( pRetimingClient );
  RetimingClient_Destroy( pRetimingClient );
  free( FilteredSubjects );
  return 0;
}

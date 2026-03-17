/*==========================================================================*
 *==========================================================================*
 * Vicon provide the software below “as is," and you use the software at your
 * own risk. Vicon makes no warranties as to performance, merchantability, 
 * fitness for a particular purpose, or any other warranties whether expressed 
 * or implied. No oral or written communication from or information provided 
 * by Vicon shall create a warranty. Under no circumstances shall Vicon be 
 * liable for direct, indirect, special, incidental, or consequential damages 
 * resulting from the use, misuse, or inability to use this software, even if 
 * Vicon has been advised of the possibility of such damages.
 *==========================================================================*
 *==========================================================================*/

/*==========================================================================*
 *==========================================================================*
 * This example will
 *     - Receive Data from Vicon Data Stream
 *     - Retrieve and display the position data for 2 object requested
 *==========================================================================*
 *==========================================================================*/

/* Vicon DSSDK */
#include "DataStreamClient.h"

#include <iostream>

#define S_FUNCTION_LEVEL 2
#define S_FUNCTION_NAME  TrackerMultiObject

/* total number of block parameters */
#define N_PAR   2  

/*
 * Need to include simstruc.h for the definition of the SimStruct and
 * its associated macro definitions.
 */
#include "simstruc.h"

/* status values */
#define STATUS_NEW_DATA              0x0001
#define STATUS_HAVE_OBJECT_1         0x0010
#define STATUS_HAVE_OBJECT_2         0x0100
#define STATUS_CLIENT_NOT_CONNECTED  0x1000

/*====================*
 * S-function methods *
 *====================*/

/* Function: mdlInitializeSizes ===============================================
 * Abstract:
 *    The sizes information is used by Simulink to determine the S-function
 *    block's characteristics (number of inputs, outputs, states, etc.).
 */
static void mdlInitializeSizes(SimStruct *S)
{
    int i;
    
    ssSetNumSFcnParams(S, N_PAR);  /* Number of expected parameters */
    
#if defined(MATLAB_MEX_FILE)
    if (ssGetNumSFcnParams(S) == ssGetSFcnParamsCount(S)) {
        if (ssGetErrorStatus(S) != NULL) {
            return;
        }
    } else {
        return; /* Parameter mismatch will be reported by Simulink */
    }
#endif

    /* single input port - not used */
    if (!ssSetNumInputPorts(S, 0)) return;
    
    /* 6 output ports */
    if ( !ssSetNumOutputPorts( S, 6 ) ) return;
    
    /* output 0 is the status */
    ssSetOutputPortWidth(    S, 0, 1 );
    ssSetOutputPortDataType( S, 0, SS_UINT32 );
    
    /* output 1 is the frame number */    
    ssSetOutputPortWidth(    S, 1, 1 );
    ssSetOutputPortDataType( S, 1, SS_UINT32 );

    /* output 2 is the translation values for Object 1 */
    /* output 3 is the rotation values for Object 1 */
    
    /* output 4 is the translation values for Object 2 */
    /* output 5 is the rotation values for Object 2 */
    
    for( i=2; i<=5; i++)
    {
        ssSetOutputPortWidth(    S, i, 3 );
        ssSetOutputPortDataType( S, i, SS_DOUBLE );
    }
    
    /* reserve element in the pointers vector, we use this to
     * persist the DSSDK client object */    
    ssSetNumPWork( S, 1 ); 
    
    /* sample times */
    ssSetNumSampleTimes( S, 1 );
    
    /* specify the sim state compliance to be same as a built-in block */
    ssSetSimStateCompliance( S, USE_DEFAULT_SIM_STATE );

    /* options */
    ssSetOptions( S,
                  SS_OPTION_WORKS_WITH_CODE_REUSE |
                  SS_OPTION_EXCEPTION_FREE_CODE |
                  SS_OPTION_USE_TLC_WITH_ACCELERATOR );
}

/* Function: mdlInitializeSampleTimes =========================================
 * Abstract:
 *    This function is used to specify the sample time(s) for your
 *    S-function. You must register the same number of sample times as
 *    specified in ssSetNumSampleTimes.
 */
static void mdlInitializeSampleTimes(SimStruct *S)
{
    ssSetSampleTime( S, 0, CONTINUOUS_SAMPLE_TIME );
    ssSetOffsetTime( S, 0, FIXED_IN_MINOR_STEP_OFFSET );
    ssSetModelReferenceSampleTimeDefaultInheritance(S);
}

#define MDL_START  
/* Function: mdlStart =======================================================
 * Abstract:
 *    This function is called once at start of model execution. If you
 *    have states that should be initialized once, this is the place
 *    to do it.
 */
static void mdlStart(SimStruct *S)
{
    // persist the C++ DSSDK object in the pointers vector
    ViconDataStreamSDK::CPP::Client *pMyClient = new ViconDataStreamSDK::CPP::Client();
    pMyClient->Connect( "localhost:801" );
    pMyClient->SetStreamMode( ViconDataStreamSDK::CPP::StreamMode::ServerPush );
    pMyClient->EnableSegmentData();
      
    ssGetPWork(S)[0] = (void *) pMyClient; 
}                                            

/* Function: mdlOutputs =======================================================
 * Abstract:
 *    In this function, you compute the outputs of your S-function
 *    block.
 */
static void mdlOutputs(SimStruct *S, int_T tid)
{
    // retrieve pointer to our persisted DSSDK connection
    ViconDataStreamSDK::CPP::Client *pMyClient = (ViconDataStreamSDK::CPP::Client *) ssGetPWork(S)[0];  
    
    uint32_T FrameNumber = 0;
    uint32_T Status  = 0;
    
    uint32_T *pStatusOut   = (uint32_T *)ssGetOutputPortSignal(S,0);
    
    uint32_T *pFrameOut    = (uint32_T *)ssGetOutputPortSignal(S,1);
    
    real_T   *pTransOut1   = (real_T *)ssGetOutputPortSignal(S,2);
    real_T   *pRotOut1     = (real_T *)ssGetOutputPortSignal(S,3);
    
    real_T   *pTransOut2   = (real_T *)ssGetOutputPortSignal(S,4);
    real_T   *pRotOut2     = (real_T *)ssGetOutputPortSignal(S,5); 
 
    
    if ( !pMyClient )
    {
        Status = STATUS_CLIENT_NOT_CONNECTED;
    }
    else if ( !pMyClient->IsConnected().Connected )
    {
        Status = STATUS_CLIENT_NOT_CONNECTED;
    }
    else
    {
        if( pMyClient->GetFrame().Result == ViconDataStreamSDK::CPP::Result::Success )
        {
            // find the data for our objects
                                    
            // Object1
            {
                std::string ObjectName1 = mxArrayToString( ssGetSFcnParam( S, 0 ) );
                ViconDataStreamSDK::CPP::Output_GetSegmentGlobalTranslation _Output_GetSegmentGlobalTranslation = 
                                pMyClient->GetSegmentGlobalTranslation( ObjectName1, ObjectName1 );
                if( ( _Output_GetSegmentGlobalTranslation.Result == ViconDataStreamSDK::CPP::Result::Success ) &&
                    ( !_Output_GetSegmentGlobalTranslation.Occluded ) )
                {
                    ViconDataStreamSDK::CPP::Output_GetSegmentGlobalRotationEulerXYZ _Output_GetSegmentGlobalRotationEulerXYZ = 
                                  pMyClient->GetSegmentGlobalRotationEulerXYZ( ObjectName1, ObjectName1 );
                    if( ( _Output_GetSegmentGlobalRotationEulerXYZ.Result == ViconDataStreamSDK::CPP::Result::Success ) &&
                        ( !_Output_GetSegmentGlobalRotationEulerXYZ.Occluded ) )
                    {
                        Status |= STATUS_HAVE_OBJECT_1;

                        // desired output is meters
                        pTransOut1[0] = _Output_GetSegmentGlobalTranslation.Translation[ 0 ] / 1000.0;
                        pTransOut1[1] = _Output_GetSegmentGlobalTranslation.Translation[ 1 ] / 1000.0;
                        pTransOut1[2] = _Output_GetSegmentGlobalTranslation.Translation[ 2 ] / 1000.0;

                        pRotOut1[0] = _Output_GetSegmentGlobalRotationEulerXYZ.Rotation[ 0 ];
                        pRotOut1[1] = _Output_GetSegmentGlobalRotationEulerXYZ.Rotation[ 1 ];
                        pRotOut1[2] = _Output_GetSegmentGlobalRotationEulerXYZ.Rotation[ 2 ];                      
                    }
                }
            }
            
             // Object2
            {
                std::string ObjectName2 = mxArrayToString( ssGetSFcnParam( S, 1 ) );
                ViconDataStreamSDK::CPP::Output_GetSegmentGlobalTranslation _Output_GetSegmentGlobalTranslation = 
                                pMyClient->GetSegmentGlobalTranslation( ObjectName2, ObjectName2 );
                if( ( _Output_GetSegmentGlobalTranslation.Result == ViconDataStreamSDK::CPP::Result::Success ) &&
                    ( !_Output_GetSegmentGlobalTranslation.Occluded ) )
                {
                    ViconDataStreamSDK::CPP::Output_GetSegmentGlobalRotationEulerXYZ _Output_GetSegmentGlobalRotationEulerXYZ = 
                                  pMyClient->GetSegmentGlobalRotationEulerXYZ( ObjectName2, ObjectName2 );
                    if( ( _Output_GetSegmentGlobalRotationEulerXYZ.Result == ViconDataStreamSDK::CPP::Result::Success ) &&
                        ( !_Output_GetSegmentGlobalRotationEulerXYZ.Occluded ) )
                    {
                        Status |= STATUS_HAVE_OBJECT_2;

                        // desired output is meters
                        pTransOut2[0] = _Output_GetSegmentGlobalTranslation.Translation[ 0 ] / 1000.0;
                        pTransOut2[1] = _Output_GetSegmentGlobalTranslation.Translation[ 1 ] / 1000.0;
                        pTransOut2[2] = _Output_GetSegmentGlobalTranslation.Translation[ 2 ] / 1000.0;

                        pRotOut2[0] = _Output_GetSegmentGlobalRotationEulerXYZ.Rotation[ 0 ];
                        pRotOut2[1] = _Output_GetSegmentGlobalRotationEulerXYZ.Rotation[ 1 ];
                        pRotOut2[2] = _Output_GetSegmentGlobalRotationEulerXYZ.Rotation[ 2 ];                      
                    }
                }      
            }
            
            ViconDataStreamSDK::CPP::Output_GetFrameNumber _Output_GetFrameNumber = pMyClient->GetFrameNumber();
            FrameNumber = _Output_GetFrameNumber.FrameNumber;
        }
    }
    
    *pStatusOut = Status;
    *pFrameOut = FrameNumber;
            
    UNUSED_ARG(tid); 
}                                                

/* Function: mdlTerminate =====================================================
 * Abstract:
 *    In this function, you should perform any actions that are necessary
 *    at the termination of a simulation.  For example, if memory was
 *    allocated in mdlStart, this is the place to free it.
 */
static void mdlTerminate(SimStruct *S)
{
    ViconDataStreamSDK::CPP::Client *pMyClient = (ViconDataStreamSDK::CPP::Client *) ssGetPWork(S)[0]; 
    if( pMyClient )
    {
        pMyClient->Disconnect();
        delete pMyClient;
    }
}                                            
/*======================================================*
 * See sfuntmpl.doc for the optional S-function methods *
 *======================================================*/

/*=============================*
 * Required S-function trailer *
 *=============================*/

#ifdef  MATLAB_MEX_FILE    /* Is this file being compiled as a MEX-file? */
#include "simulink.c"      /* MEX-file interface mechanism */
#else
#include "cg_sfun.h"       /* Code generation registration function */
#endif


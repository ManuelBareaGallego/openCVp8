/////////////////////////////////////////////////////////
//  Dpto ATC. 
//  version 2. Nov. 2012. Fernando Diaz del Rio/Pablo Iñigo Blasco /Manuel J Dominguez Morales
/////////////////////////////////////////////////////////
// this timing library is based on C++ OpenCV timing function getCPUTickCount(), that are
// based on Windows x86 QueryPerformanceCounter library  .

// QueryPerformanceCounter  Granularity for x86 usually is 0.5 us approx.
//but here the CPU frequency must be defined in QueryPerformanceTiming_rdtsc.h

//	Note: getCPUTickCount() and rdtsc x86 instructions (time stamp counter) is finer but it depends strongly 
//  on current CPU frequency (that is variable in most current CPUs ), and measures
//  are not exact at all if CPU frequency does not match stock CPU (e.g if CPU load is low )
/////////////////////////////////////////////////////////


#include <stdio.h>
#include <intrin.h>
#include <iostream>

using namespace std;

#include "processor_power_information.h"
#include "QueryPerformanceTiming_rdtsc.h"


/////////////////////////////////////////////////////////
//public methods 
/////////////////////////////////////////////////////////
QPTimerRDTSC::QPTimerRDTSC()
{
	LARGE_INTEGER qfrec;

	minimumTime= 0xFFFFFFFFFFFFFFFF;
	times=0;
	overhead= 0;
	totalTime=0;
	Reset();
	QueryPerformanceFrequency(&qfrec);
	QPFrequencyHz = qfrec.QuadPart;
	DisableOutput=false;
}

void QPTimerRDTSC::Start()
{
	ReadCycles(&startTime);
}

void QPTimerRDTSC::Stop()
{
	uint64_t now;
	ReadCycles(&now);

	uint64_t r;
	RestCycles(&r,&now,&startTime);
	RestCycles(&r,&r,&overhead);

	AddCycles(&result,&result,&r);

	AddCycles(&totalTime,&totalTime,&r);

	if (minimumTime > result)
			minimumTime = result;

	times++;
}
///////////////////////////////////////////////////
void QPTimerRDTSC::Calibrate()
{
	overhead= 0xFFFFFFFFFFFFFFFF;

	for (int i=0; i<TEST_TIMES; i++)
	{
		uint64_t r;
		uint64_t now;

		ReadCycles(&startTime);
		ReadCycles(&now);

		RestCycles(&r,&now,&startTime);
		
		if (overhead > r)
			overhead = r;
	}
	ResetAll();
}


void QPTimerRDTSC::Reset()
{
	//startTime=0.0;
	result=0;
}

void QPTimerRDTSC::ResetAll()
{
	//startTime=0.0;
	result=0;
	minimumTime=0xFFFFFFFFFFFFFFFF;
	totalTime=0;

	times=0;
}

void QPTimerRDTSC:: PrintTime(const char* message)
{
	if(!DisableOutput)
		PrintTimeSeconds(message,&result);
}

void QPTimerRDTSC:: PrintCycles(const char* message)
{
	if(!DisableOutput)
		PrintCycles(message,&result);
}

void QPTimerRDTSC::PrintMinimumTime(const char* message)
{
	if(!DisableOutput)
		PrintTimeSeconds(message,&minimumTime);
}

void QPTimerRDTSC::PrintMinimumCycles(const char* message)
{
	if(!DisableOutput)
		PrintCycles(message,&minimumTime);
}

void QPTimerRDTSC::PrintMinimumCyclesByIteration(const char* message, int num)
{
	if(!DisableOutput)
		PrintCyclesByIteration(message,&minimumTime,num);
}

void QPTimerRDTSC::PrintMeanTime(const char* message)
{
	if(!DisableOutput)
	{
		uint64_t MeanTime ;
		MeanTime = (totalTime/times);

		PrintTimeSeconds(message,&MeanTime);
		cout << "   (in " << times << " measures)" <<endl;
	}
}

void QPTimerRDTSC::PrintMeanCycles(const char* message)
{
	if(!DisableOutput)
	{
		uint64_t MeanCycles ;
		MeanCycles = (totalTime/times);

		PrintCycles(message,&MeanCycles );
		cout << "   (in " << times << " measures)" <<endl;
	}
}
int QPTimerRDTSC::NumberOfMeasures(void)
{
	return times;
}

/////////////////////////////////////////////////////////
//private methods 
/////////////////////////////////////////////////////////

void QPTimerRDTSC::ReadCycles (uint64_t *cycles)
{
	unsigned int dummy;

	*cycles = __rdtscp(&dummy);
}
//void 
//	QPTimer::ReadCycles (double *cycles)
//{
//	// No serializing instruction: there is no flush of pipeline processor
//	// OpenCV timing. It uses QueryPerformanceCounter. Granularity for x86 usually is 0.5 us approx.
////	*cycles = (double)getTickCount();
//	*cycles = (double)getCPUTickCount();
//	
//}

void QPTimerRDTSC::PrintCycles (const char *pc, uint64_t *cycles)
{
	cout << pc << *cycles << endl;
}	


void QPTimerRDTSC::PrintCyclesByIteration (const char *pc, uint64_t *cycles, int NumberOfIterations)
{
	uint64_t hz = GetCurrentCPUFreq();
	cout << pc << *cycles/(double)NumberOfIterations << " (@ " << hz / 1000000000.0 << " GHz)" << endl;
}	

		

void QPTimerRDTSC::PrintTimeSeconds (const char *pc, uint64_t *cycles)
{   
	cout << pc << (double)(*cycles)/QPFrequencyHz << endl;
/*	if(pc!=NULL && strlen(pc)>0)
		printf("%s",pc);		*/
}
void QPTimerRDTSC::RestCycles (uint64_t *c_dif, uint64_t *c_final, uint64_t *c_init)
{
	*c_dif=*c_final-*c_init;

}

void QPTimerRDTSC::AddCycles (uint64_t *c_add, uint64_t *c_final, uint64_t *c_init)
{
	*c_add=*c_final+*c_init;
}

uint64_t QPTimerRDTSC::GetCurrentCPUFreq(void)
{
	PROCESSOR_POWER_INFORMATION *powerinfo;
	SYSTEM_INFO sinfo;
	uint64_t hz;

	GetSystemInfo(&sinfo);
	powerinfo = (PROCESSOR_POWER_INFORMATION *)malloc(sizeof *powerinfo * sinfo.dwNumberOfProcessors);
	CallNtPowerInformation(ProcessorInformation, NULL, 0, powerinfo, sizeof *powerinfo * sinfo.dwNumberOfProcessors);
	hz = powerinfo->CurrentMhz * 1000000;
	free(powerinfo);
	return hz;
}
/////////////////////////////////////////////////////////
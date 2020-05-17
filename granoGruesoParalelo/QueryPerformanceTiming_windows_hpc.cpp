/////////////////////////////////////////////////////////
//  Dpto ATC. 
//  version 2. Nov. 2012. Fernando Diaz del Rio/Pablo Iñigo Blasco /Manuel J Dominguez Morales
//  Updated: RDTSC no longer reflects true CPU frequency. Resorting to QueryPerformanceCounter
//           and NtGetPowerInformation to retrieve current true CPU frequency
//           Keeping 64 bit variables instead of converting to double
//           March. 2020 (corona virus month) Miguel Angel Rodriguez Jodar
/////////////////////////////////////////////////////////
// this timing library is based on C++ OpenCV timing function getCPUTickCount(), that are
// based on Windows x86 QueryPerformanceCounter library  .
/////////////////////////////////////////////////////////


#include <stdio.h>
#include <iostream>

using namespace std;

#include "processor_power_information.h"
#include "QueryPerformanceTiming_windows_hpc.h"

/////////////////////////////////////////////////////////
//public methods 
/////////////////////////////////////////////////////////
QPTimerHPC::QPTimerHPC()
{
	LARGE_INTEGER qfrec;

	minimumTime = 0xFFFFFFFFFFFFFFFF;
	times = 0;
	overhead = 0;
	totalTime = 0;
	DisableOutput = false;
	QueryPerformanceFrequency(&qfrec);
	qpfrequency = qfrec.QuadPart;
	Reset();
}

void QPTimerHPC::Start()
{
	ReadCycles(&startTime);
}

void QPTimerHPC::Stop()
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
void QPTimerHPC::Calibrate()
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


void QPTimerHPC::Reset()
{
	result=0;
}

void QPTimerHPC::ResetAll()
{
	result=0;
	minimumTime=0xFFFFFFFFFFFFFFFF;
	totalTime=0;

	times=0;
}

void QPTimerHPC:: PrintTime(const char* message)
{
	if(!DisableOutput)
		PrintTimeSeconds(message,&result);
}

void QPTimerHPC::PrintCycles(const char* message)
{
	if(!DisableOutput)
		PrintCycles(message, &result);
}

void QPTimerHPC::PrintMinimumTime(const char* message)
{
	if(!DisableOutput)
		PrintTimeSeconds(message,&minimumTime);
}

void QPTimerHPC::PrintMinimumCycles(const char* message)
{
	if(!DisableOutput)
		PrintCycles(message, &minimumTime);
}

void QPTimerHPC::PrintMinimumCyclesByIteration(const char* message, int num)
{
	if(!DisableOutput)
		PrintCyclesByIteration(message,&minimumTime,num);
}

void QPTimerHPC::PrintMeanTime(const char* message)
{
	if(!DisableOutput)
	{
		uint64_t MeanTime ;
		MeanTime = (totalTime/times);

		PrintTimeSeconds(message, &MeanTime);
		cout << "   (in " << times << " measures)" <<endl;
	}
}

void QPTimerHPC::PrintMeanCycles(const char* message)
{
	if(!DisableOutput)
	{
		uint64_t MeanCycles ;
		MeanCycles = (totalTime/times);

		PrintCycles(message, &MeanCycles);
		cout << "   (in " << times << " measures)" <<endl;
	}
}
int QPTimerHPC::NumberOfMeasures(void)
{
	return times;
}

/////////////////////////////////////////////////////////
//private methods 
/////////////////////////////////////////////////////////

void QPTimerHPC::ReadCycles (uint64_t *cycles)
{
	LARGE_INTEGER time;

	QueryPerformanceCounter(&time);
	*cycles = time.QuadPart;	
}

void QPTimerHPC::PrintCycles (const char *pc, uint64_t *cycles)
{
	uint64_t hz = GetCurrentCPUFreq();
	cout.precision(4);
	cout << pc << fixed << (*cycles)*hz / (double)qpfrequency << " (@ " << hz / 1000000000.0 << " GHz)" << endl;
}

void QPTimerHPC::PrintCyclesByIteration (const char *pc, uint64_t *cycles, int NumberOfIterations)
{
	uint64_t hz = GetCurrentCPUFreq();
	cout.precision(4);
	cout << pc << fixed << (*cycles)*hz/(qpfrequency*(double)NumberOfIterations) << " (@ " << hz/1000000000.0 << " GHz)" << endl;
}	

void QPTimerHPC::PrintTimeSeconds (const char *pc, uint64_t *cycles)
{   
	cout << pc << (double)(*cycles)/qpfrequency << endl;
}

void QPTimerHPC::RestCycles (uint64_t *c_dif, uint64_t *c_final, uint64_t *c_init)
{
	*c_dif=*c_final-*c_init;
}

void QPTimerHPC::AddCycles (uint64_t *c_add, uint64_t *c_final, uint64_t *c_init)
{
	*c_add=*c_final+*c_init;
}

uint64_t QPTimerHPC::GetCurrentCPUFreq(void)
{
	PROCESSOR_POWER_INFORMATION *powerinfo;
	SYSTEM_INFO sinfo;
	uint64_t hz;

	GetSystemInfo(&sinfo);
	powerinfo = (PROCESSOR_POWER_INFORMATION *)malloc (sizeof *powerinfo * sinfo.dwNumberOfProcessors);
	CallNtPowerInformation(ProcessorInformation, NULL, 0, powerinfo, sizeof *powerinfo * sinfo.dwNumberOfProcessors);
	hz = powerinfo->CurrentMhz * 1000000;
	free(powerinfo);
	return hz;
}
/////////////////////////////////////////////////////////
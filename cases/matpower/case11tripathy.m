function mpc = case11tripathy
%CASE11TRIPATHY  11-bus ill-conditioned test system (Tripathy 1982 lineage).
%   Transcribed from Tables 7.1-7.3 of the "Voltage Behaviour" textbook
%   chapter (ndl.ethernet.edu.et/bitstream/123456789/30637/1/28.pdf),
%   which derived branch data from the bus admittance matrix given in
%   Tripathy, Prasad, Malik, Hope, "Load-Flow Solutions for Ill-Conditioned
%   Power Systems by a Newton-Like Method", IEEE TPAS-101(10), 1982,
%   assuming unit transformer ratios; half-line charging 0.0001 pu per end.
%   Base 100 MVA.
%
%   Literature conflict on solvability at nominal load:
%   - Iwamoto & Tamura 1981: solution does not exist (mu* -> 0, |J| -> 0);
%   - textbook: "reported to be a working system", cites MLF voltages;
%   - Bonini Neto et al. 2015 (CPF): MLP lambda = 0.69949 < 1.

mpc.version = '2';
mpc.baseMVA = 100;

%% bus data
%	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
mpc.bus = [
	1	3	0	0	0	0	1	1.0	0	100	1	1.1	0.9;
	2	1	0	0	0	0	1	1.0	0	100	1	1.1	0.9;
	3	1	12.8	6.2	0	0	1	1.0	0	100	1	1.1	0.9;
	4	1	0	0	0	0	1	1.0	0	100	1	1.1	0.9;
	5	1	16.5	8.0	0	0	1	1.0	0	100	1	1.1	0.9;
	6	1	9.0	6.8	0	0	1	1.0	0	100	1	1.1	0.9;
	7	1	0	0	0	0	1	1.0	0	100	1	1.1	0.9;
	8	1	0	0	0	0	1	1.0	0	100	1	1.1	0.9;
	9	1	2.6	0.9	0	0	1	1.0	0	100	1	1.1	0.9;
	10	1	0	0	0	0	1	1.0	0	100	1	1.1	0.9;
	11	1	15.8	5.7	0	0	1	1.0	0	100	1	1.1	0.9;
];

%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf
mpc.gen = [
	1	60	27.6	999	-999	1.0	100	1	999	0	0	0	0	0	0	0	0	0	0	0	0;
];

%% branch data
%	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
mpc.branch = [
	1	2	0.0	0.0707	0.0002	0	0	0	0	0	1	-360	360;
	2	3	0.0	0.1540	0.0002	0	0	0	0	0	1	-360	360;
	2	4	0.0377	0.0413	0.0002	0	0	0	0	0	1	-360	360;
	3	5	0.1228	0.1803	0.0002	0	0	0	0	0	1	-360	360;
	4	5	0.0	0.4593	0.0002	0	0	0	0	0	1	-360	360;
	4	6	0.0	0.0176	0.0002	0	0	0	0	0	1	-360	360;
	4	7	0.6114	0.8118	0.0002	0	0	0	0	0	1	-360	360;
	7	8	0.1621	0.2167	0.0002	0	0	0	0	0	1	-360	360;
	8	9	0.0718	0.7180	0.0002	0	0	0	0	0	1	-360	360;
	8	10	0.4098	0.5600	0.0002	0	0	0	0	0	1	-360	360;
	10	11	0.0246	0.2646	0.0002	0	0	0	0	0	1	-360	360;
];

%% generator cost data (dummy)
%	model	startup	shutdown	n	c2	c1	c0
mpc.gencost = [
	2	0	0	3	0	0	0;
];

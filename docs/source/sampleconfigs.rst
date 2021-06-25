Sample Configurations for h5bench_patterns
==============================================

-------------------------------------
sample_2d_compressed.cfg
-------------------------------------

.. code-block::

	# this is a comment
	# Benchmark mode can only be one of these: CC/CI/IC/II/CC2D/CI2D/IC2D/II2D/CC2D/CC3D
	IO_OPERATION=READ
	MEM_PATTERN=CONTIG
	FILE_PATTERN=CONTIG
	READ_OPTION=FULL
	NUM_DIMS=2
	NUM_PARTICLES=8 M
	TIMESTEPS=5
	EMULATED_COMPUTE_TIME_PER_TIMESTEP=1 s
	DIM_1=4096
	DIM_2=2048
	DIM_3=1
	COMPRESS=YES  # to enable parallel compression(chunking)
	CHUNK_DIM_1=512  # chunk dimensions
	CHUNK_DIM_2=256
	CHUNK_DIM_3=1


-------------------------------------
sample_read_cc1d.cfg
-------------------------------------

.. code-block::

	# this is a comment
	# Benchmark mode can only be one of these: CC/CI/IC/II/CC2D/CI2D/IC2D/II2D/CC2D/CC3D
	IO_OPERATION=READ
	READ_OPTION=FULL
	NUM_DIMS=1
	NUM_PARTICLES=8 M
	TIMESTEPS=5
	EMULATED_COMPUTE_TIME_PER_TIMESTEP=1 s
	DIM_1=8 M
	DIM_2=1
	DIM_3=1


-------------------------------------
sample_read_cc1d_es1.cfg
-------------------------------------

.. code-block::

	# this is a comment
	IO_OPERATION=READ
	TO_READ_NUM_PARTICLES=16777216
	READ_OPTION=READ_FULL
	MEM_PATTERN=CONTIG
	FILE_PATTERN=CONTIG
	TIMESTEPS=5
	DELAYED_CLOSE_TIMESTEPS=2
	COLLECTIVE_DATA=NO #Optional, default for NO.
	COLLECTIVE_METADATA=NO #Optional, default for NO.
	EMULATED_COMPUTE_TIME_PER_TIMESTEP=1 s
	NUM_DIMS=1
	DIM_1=16777216 # 16777216, 8388608
	DIM_2=1
	DIM_3=1
	IO_MEM_LIMIT=1 G
	ASYNC_MODE=EXP  #NON
	#CSV_FILE=perf_read_1d.csv
	#==================================


-------------------------------------
sample_read_cc2d.cfg
-------------------------------------

.. code-block::

	# this is a comment
	# Benchmark mode can only be one of these: CC/CI/IC/II/CC2D/CI2D/IC2D/II2D/CC2D/CC3D
	IO_OPERATION=READ
	MEM_PATTERN=CONTIG
	FILE_PATTERN=CONTIG
	READ_OPTION=FULL
	NUM_DIMS=2
	NUM_PARTICLES=8 M
	TIMESTEPS=5
	EMULATED_COMPUTE_TIME_PER_TIMESTEP=1 s
	DIM_1=2048
	DIM_2=4096
	DIM_3=1

-------------------------------------
sample_read_strided.cfg
-------------------------------------

.. code-block::

	# this is a comment
	# Benchmark mode can only be one of these: CC/CI/IC/II/CC2D/CI2D/IC2D/II2D/CC2D/CC3D
	IO_OPERATION=READ
	TO_READ_NUM_PARTICLES=16777216
	READ_OPTION=FULL
	MEM_PATTERN=CONTIG
	FILE_PATTERN=STRIDED
	TIMESTEPS=5
	DELAYED_CLOSE_TIMESTEPS=10
	COLLECTIVE_DATA=NO #Optional, default for NO.
	COLLECTIVE_METADATA=NO #Optional, default for NO.
	EMULATED_COMPUTE_TIME_PER_TIMESTEP=1 s
	NUM_DIMS=1
	DIM_1=16777216 # 16777216, 8388608
	DIM_2=1
	DIM_3=1
	STRIDE_SIZE=64
	BLOCK_SIZE=16
	BLOCK_CNT=128
	ASYNC_MODE=NO  #NON
	CSV_FILE=perf_read_1d.csv
	#==================================


-------------------------------------
sample_write_cc1d.cfg
-------------------------------------

.. code-block::

	# this is a comment
	# Benchmark mode can only be one of these: CC/CI/IC/II/CC2D/CI2D/IC2D/II2D/CC2D/CC3D
	# Template cof include all options
	IO_OPERATION=WRITE
	MEM_PATTERN=CONTIG
	FILE_PATTERN=CONTIG
	NUM_PARTICLES=16 M #16 K/G
	TIMESTEPS=5
	#IO_OPERATION=READ #WRITE
	#MEM_PATTERN=CONTIG #INTERLEAVED STRIDED
	#FILE_PATTERN=CONTIG #STRIDED
	DELAYED_CLOSE_TIMESTEPS=2
	COLLECTIVE_DATA=NO #Optional, default for NO.
	COLLECTIVE_METADATA=NO #Optional, default for NO.
	EMULATED_COMPUTE_TIME_PER_TIMESTEP=1 s #1 ms, 1 min 
	NUM_DIMS=1
	DIM_1=16777216 #16777216 # 16777216, 8388608
	DIM_2=1
	DIM_3=1
	ASYNC_MODE=NON #EXP #ASYNC_IMP ASYNC_NON ASYNC_EXP
	CSV_FILE=perf_write_1d.csv
	#===========================
	#WRITE_PATTERN=CC


-------------------------------------
sample_write_cc1d_es1.cfg
-------------------------------------

.. code-block::

	# this is a comment
	IO_OPERATION=WRITE
	MEM_PATTERN=CONTIG
	FILE_PATTERN=CONTIG
	NUM_PARTICLES=16 M #K, M, G
	TIMESTEPS=5
	DELAYED_CLOSE_TIMESTEPS=2
	COLLECTIVE_DATA=NO
	#Optional, default for NO.
	COLLECTIVE_METADATA=NO
	#Optional, default for NO.
	EMULATED_COMPUTE_TIME_PER_TIMESTEP=1 s
	#1 ms, 1 min 
	NUM_DIMS=1
	DIM_1=16777216
	#16777216 # 16777216, 8388608
	DIM_2=1
	DIM_3=1
	IO_MEM_LIMIT=1 G
	#ASYNC_MODE=ASYNC_EXP
	ASYNC_MODE=EXP #IMP NON EXP
	#CSV_FILE=perf_write_1d.csv
	#===========================
	#WRITE_PATTERN=CC

-------------------------------------
sample_write_cc1d_fileperproc.cfg
-------------------------------------

.. code-block::

	# this is a comment
	# Benchmark mode can only be one of these: CC/CI/IC/II/CC2D/CI2D/IC2D/II2D/CC2D/CC3D
	WRITE_PATTERN=CC
	PARTICLE_CNT_M=8
	TIME_STEPS_CNT=1
	DATA_COLL=NO #Optional, default for NO.
	META_COLL=NO #Optional, default for NO.
	SLEEP_TIME=1
	DIM_1=8388608
	DIM_2=1
	DIM_3=1
	ASYNC_MODE=ASYNC_NON
	CSV_FILE=perf_write_1d.csv
	FILE_PER_PROC=YES #Optional, default is NO.

-------------------------------------
sample_write_cc2d.cfg
-------------------------------------

.. code-block::

	# this is a comment
	# Benchmark mode can only be one of these: CC/CI/IC/II/CC2D/CI2D/IC2D/II2D/CC2D/CC3D
	# Template cof include all options
	IO_OPERATION=WRITE
	MEM_PATTERN=CONTIG
	FILE_PATTERN=CONTIG
	NUM_PARTICLES=16 M #16 K/G
	TIMESTEPS=5
	#IO_OPERATION=READ #WRITE
	#MEM_PATTERN=CONTIG #INTERLEAVED STRIDED
	#FILE_PATTERN=CONTIG #STRIDED
	DELAYED_CLOSE_TIMESTEPS=2
	COLLECTIVE_DATA=NO #Optional, default for NO.
	COLLECTIVE_METADATA=NO #Optional, default for NO.
	EMULATED_COMPUTE_TIME_PER_TIMESTEP=1 s #1 ms, 1 min 
	NUM_DIMS=2
	DIM_1=4096 #16777216 # 16777216, 8388608
	DIM_2=4096
	DIM_3=1
	ASYNC_MODE=NON #EXP #ASYNC_IMP ASYNC_NON ASYNC_EXP
	CSV_FILE=perf_write_1d.csv
	#===========================
	#WRITE_PATTERN=CC

-------------------------------------
sample_write_strided.cfg
-------------------------------------

.. code-block::

	# this is a comment
	# Benchmark mode can only be one of these: CC/CI/IC/II/CC2D/CI2D/IC2D/II2D/CC2D/CC3D
	WRITE_PATTERN=CC
	NUM_PARTICLES=16
	TIMESTEPS=1
	COLLECTIVE_DATA=NO #Optional, default for NO.
	COLLECTIVE_METADATA=NO #Optional, default for NO.
	EMULATED_COMPUTE_TIME_PER_TIMESTEP=1
	DIM_1=8388608
	DIM_2=1
	DIM_3=1
	STRIDE_SIZE=2
	BLOCK_SIZE=2
	BLOCK_CNT=1048576


-------------------------------------
template.cfg
-------------------------------------

.. code-block::

	#========================================================
	#   General settings
	NUM_PARTICLES=16 M # 16 K  16777216
	TIMESTEPS=5
	EMULATED_COMPUTE_TIME_PER_TIMESTEP=1 s #1 ms, 1 min
	#========================================================
	#   Benchmark data dimensionality
	NUM_DIMS=1
	DIM_1=16777216 # 16777216, 16 M
	DIM_2=1
	DIM_3=1
	#========================================================
	#   IO pattern settings
	IO_OPERATION=READ # WRITE
	MEM_PATTERN=CONTIG # INTERLEAVED STRIDED
	FILE_PATTERN=CONTIG # STRIDED
	#========================================================
	#    Options for IO_OPERATION=READ
	READ_OPTION=FULL # PARTIAL STRIDED
	TO_READ_NUM_PARTICLES=4 M
	#========================================================
	#    Strided access parameters
	#STRIDE_SIZE=
	#BLOCK_SIZE=
	#BLOCK_CNT=
	#========================================================
	# Collective data/metadata settings
	#COLLECTIVE_DATA=NO #Optional, default for NO.
	#COLLECTIVE_METADATA=NO #Optional, default for NO.
	#========================================================
	#    Compression, optional, default is NO.
	#COMPRESS=NO
	#CHUNK_DIM_1=1
	#CHUNK_DIM_2=1
	#CHUNK_DIM_3=1
	#========================================================
	#    Async related settings
	DELAYED_CLOSE_TIMESTEPS=2
	IO_MEM_LIMIT=5000 K
	ASYNC_MODE=EXP #EXP NON
	#========================================================
	#    Output performance results to a CSV file
	#CSV_FILE=perf_write_1d.csv
	#    
	#FILE_PER_PROC=


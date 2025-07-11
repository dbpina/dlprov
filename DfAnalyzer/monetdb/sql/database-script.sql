START TRANSACTION;

SET SCHEMA "public";

-- sequences
CREATE SEQUENCE "df_id_seq" as integer START WITH 1;
CREATE SEQUENCE "version_id_seq" as integer START WITH 1;
CREATE SEQUENCE "dt_id_seq" as integer START WITH 1;
CREATE SEQUENCE "program_id_seq" as integer START WITH 1;
CREATE SEQUENCE "ds_id_seq" as integer START WITH 1;
CREATE SEQUENCE "dd_id_seq" as integer START WITH 1;
CREATE SEQUENCE "extractor_id_seq" as integer START WITH 1;
CREATE SEQUENCE "ecombination_id_seq" as integer START WITH 1;
CREATE SEQUENCE "att_id_seq" as integer START WITH 1;
CREATE SEQUENCE "task_id_seq" as integer START WITH 1;
CREATE SEQUENCE "exec_id_seq" as integer START WITH 1;
CREATE SEQUENCE "file_id_seq" as integer START WITH 1;
CREATE SEQUENCE "performance_id_seq" as integer START WITH 1;
CREATE SEQUENCE "user_id_seq" as integer START WITH 1;
CREATE SEQUENCE "hardware_id_seq" as integer START WITH 1;


-- tables
CREATE TABLE dataflow(
	id INTEGER DEFAULT NEXT VALUE FOR "public"."df_id_seq" NOT NULL,
	uuid VARCHAR(36),
	tag VARCHAR(50) NOT NULL,
	PRIMARY KEY ("id")
);

CREATE TABLE dataflow_version(
	version INTEGER DEFAULT NEXT VALUE FOR "public"."version_id_seq" NOT NULL,
	df_id INTEGER NOT NULL,
	uuid VARCHAR(36),
	PRIMARY KEY ("version"),
	FOREIGN KEY ("df_id") REFERENCES dataflow("id") ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE data_transformation(
	id INTEGER DEFAULT NEXT VALUE FOR "public"."dt_id_seq" NOT NULL,
	uuid VARCHAR(36),
	df_id INTEGER NOT NULL,
	tag VARCHAR(50) NOT NULL,
	PRIMARY KEY ("id"),
	FOREIGN KEY ("df_id") REFERENCES dataflow("id") ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE program(
	id INTEGER DEFAULT NEXT VALUE FOR "public"."program_id_seq" NOT NULL,
	uuid VARCHAR(36),	
	df_id INTEGER NOT NULL,
	name VARCHAR(200) NOT NULL,
	path VARCHAR(500) NOT NULL,
	PRIMARY KEY ("id"),
	FOREIGN KEY ("df_id") REFERENCES dataflow("id") ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE use_program(
	dt_id INTEGER NOT NULL,
	program_id INTEGER NOT NULL,
	uuid VARCHAR(36),
	PRIMARY KEY ("dt_id","program_id"),
	FOREIGN KEY ("dt_id") REFERENCES data_transformation("id") ON DELETE CASCADE ON UPDATE CASCADE,
	FOREIGN KEY ("program_id") REFERENCES program("id") ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE data_set(
	id INTEGER DEFAULT NEXT VALUE FOR "public"."ds_id_seq" NOT NULL,
	uuid VARCHAR(36),
	df_id INTEGER NOT NULL,
	tag VARCHAR(50) NOT NULL,
	PRIMARY KEY ("id"),
	FOREIGN KEY ("df_id") REFERENCES dataflow("id") ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE data_dependency(
	id INTEGER DEFAULT NEXT VALUE FOR "public"."dd_id_seq" NOT NULL,
	uuid VARCHAR(36),	
	previous_dt_id INTEGER,
	next_dt_id INTEGER,
	ds_id INTEGER NOT NULL,
	PRIMARY KEY ("id"),
	FOREIGN KEY ("previous_dt_id") REFERENCES data_transformation("id") ON DELETE CASCADE ON UPDATE CASCADE,
	FOREIGN KEY ("next_dt_id") REFERENCES data_transformation("id") ON DELETE CASCADE ON UPDATE CASCADE,
	FOREIGN KEY ("ds_id") REFERENCES data_set("id") ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE extractor(
	id INTEGER DEFAULT NEXT VALUE FOR "public"."extractor_id_seq" NOT NULL,
	uuid VARCHAR(36),	
	ds_id INTEGER NOT NULL,
	tag VARCHAR(20) NOT NULL,
	cartridge VARCHAR(20) NOT NULL,
	extension VARCHAR(20) NOT NULL,
	PRIMARY KEY ("id"),
	FOREIGN KEY ("ds_id") REFERENCES data_set("id") ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE extractor_combination(
	id INTEGER DEFAULT NEXT VALUE FOR "public"."ecombination_id_seq" NOT NULL,
	uuid VARCHAR(36),	
	ds_id INTEGER NOT NULL,
	outer_ext_id INTEGER NOT NULL,
	inner_ext_id INTEGER NOT NULL,
	keys VARCHAR(100) NOT NULL,
	key_types VARCHAR(100) NOT NULL,
	PRIMARY KEY ("id"),
	FOREIGN KEY ("ds_id") REFERENCES data_set("id") ON DELETE CASCADE ON UPDATE CASCADE,
	FOREIGN KEY ("outer_ext_id") REFERENCES extractor("id") ON DELETE CASCADE ON UPDATE CASCADE,
	FOREIGN KEY ("inner_ext_id") REFERENCES extractor("id") ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE attribute(
	id INTEGER DEFAULT NEXT VALUE FOR "public"."att_id_seq" NOT NULL,
	uuid VARCHAR(36),	
	ds_id INTEGER NOT NULL,
	extractor_id INTEGER,
	name VARCHAR(30),
	type VARCHAR(15),
	PRIMARY KEY ("id"),
	FOREIGN KEY ("ds_id") REFERENCES data_set("id") ON DELETE CASCADE ON UPDATE CASCADE,
	FOREIGN KEY ("extractor_id") REFERENCES extractor("id") ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE data_scientist(
	id INTEGER DEFAULT NEXT VALUE FOR "public"."user_id_seq" NOT NULL,
	uuid VARCHAR(36),	
	email VARCHAR(50) NOT NULL,
	PRIMARY KEY ("id")
);

CREATE TABLE hardware_info(
	id INTEGER DEFAULT NEXT VALUE FOR "public"."hardware_id_seq" NOT NULL,
	uuid VARCHAR(36),	
	hostname VARCHAR(50) NOT NULL,
	os VARCHAR(50),  
	platform VARCHAR(50), 
	architecture VARCHAR(50), 
	processor VARCHAR(50) , 
    ram_total_gb DOUBLE, 
    ram_limit_gb DOUBLE, 
    disk_total_gb DOUBLE, 
    disk_used_gb DOUBLE, 
    disk_free_gb DOUBLE,
	gpus VARCHAR(50), 
	in_container BOOLEAN,
	PRIMARY KEY ("id")
);

CREATE TABLE dataflow_execution(
	id INTEGER DEFAULT NEXT VALUE FOR "public"."exec_id_seq" NOT NULL,
	uuid VARCHAR(36),	
	tag VARCHAR(50) NOT NULL,
	df_id INTEGER NOT NULL,
	scientist_id INTEGER,
	hardware_id INTEGER,
	PRIMARY KEY ("tag"),
	FOREIGN KEY ("df_id") REFERENCES dataflow("id") ON DELETE CASCADE ON UPDATE CASCADE,
	FOREIGN KEY ("scientist_id") REFERENCES data_scientist("id") ON DELETE CASCADE ON UPDATE CASCADE,
	FOREIGN KEY ("hardware_id") REFERENCES hardware_info("id") ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE task(
	id INTEGER DEFAULT NEXT VALUE FOR "public"."task_id_seq" NOT NULL,
	uuid VARCHAR(36),	
	identifier INTEGER NOT NULL,
	df_version INTEGER NOT NULL,
	df_exec VARCHAR(50) NOT NULL,
	dt_id INTEGER NOT NULL,
	status VARCHAR(10),
	workspace VARCHAR(500),
	computing_resource VARCHAR(100),
	output_msg TEXT,
	error_msg TEXT,
	PRIMARY KEY ("id"),
	FOREIGN KEY ("df_version") REFERENCES dataflow_version("version") ON DELETE CASCADE ON UPDATE CASCADE,
	FOREIGN KEY ("df_exec") REFERENCES dataflow_execution("tag") ON DELETE CASCADE ON UPDATE CASCADE,
	FOREIGN KEY ("dt_id") REFERENCES data_transformation("id") ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE file(
	id INTEGER DEFAULT NEXT VALUE FOR "public"."file_id_seq" NOT NULL,
	uuid VARCHAR(36),	
	task_id INTEGER NOT NULL,
	name VARCHAR(200) NOT NULL,
	path VARCHAR(500) NOT NULL,
	PRIMARY KEY ("id"),
	FOREIGN KEY ("task_id") REFERENCES task("id") ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE performance(
	id INTEGER DEFAULT NEXT VALUE FOR "public"."performance_id_seq" NOT NULL,
	uuid VARCHAR(36),	
	task_id INTEGER NOT NULL,	
	subtask_id INTEGER,	
	method VARCHAR(30) NOT NULL,
	description VARCHAR(200),
	starttime VARCHAR(30),
	endtime VARCHAR(30),
	invocation TEXT,
	PRIMARY KEY ("id"),
	FOREIGN KEY ("task_id") REFERENCES task("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- procedures
-- DROP FUNCTION insertDataflow;
CREATE FUNCTION insertDataflow (v_tag VARCHAR(50))
RETURNS INTEGER
BEGIN
	DECLARE v_df_id INTEGER;
    SELECT df.id INTO v_df_id FROM dataflow df WHERE df.tag=v_tag;
    IF(v_df_id IS NULL) THEN
    	#SELECT NEXT VALUE FOR "df_id_seq" into v_df_id;
    	INSERT INTO dataflow(tag) VALUES (v_tag);
	END IF;
	RETURN SELECT df.id FROM dataflow df WHERE df.tag=v_tag;
END;

-- DROP FUNCTION insertDataflowVersion;
CREATE FUNCTION insertDataflowVersion (vdf_id INTEGER)
RETURNS INTEGER
BEGIN
	DECLARE version_id INTEGER;
	#SELECT NEXT VALUE FOR "version_id_seq" into version_id;
    INSERT INTO dataflow_version(df_id) VALUES (vdf_id);
	RETURN SELECT dfv.version FROM dataflow_version dfv WHERE dfv.df_id=vdf_id AND dfv.version=get_value_for('public', 'version_id_seq');
END;

-- DROP FUNCTION insertDataTransformation;
CREATE FUNCTION insertDataTransformation (vdf_id INTEGER, vtag VARCHAR(50))
RETURNS INTEGER
BEGIN
	DECLARE vdt_id INTEGER;
    SELECT id INTO vdt_id FROM data_transformation WHERE df_id = vdf_id AND tag = vtag;
    IF(vdt_id IS NULL) THEN
    	#SELECT NEXT VALUE FOR "dt_id_seq" into vdt_id;
    	INSERT INTO data_transformation(df_id,tag) VALUES (vdf_id,vtag);
	END IF;
	RETURN SELECT id FROM data_transformation WHERE df_id = vdf_id AND tag = vtag;
END;

-- DROP FUNCTION insertProgram;
CREATE FUNCTION insertProgram (vdf_id INTEGER, vdt_id INTEGER, vname VARCHAR(200), vpath VARCHAR(500))
RETURNS INTEGER
BEGIN
	DECLARE vprogram_id INTEGER;
    SELECT id INTO vprogram_id FROM program p WHERE df_id = vdf_id AND name = vname AND path = vpath;

    IF(vprogram_id IS NULL) THEN
    	SELECT NEXT VALUE FOR "public"."program_id_seq" into vprogram_id;
    	INSERT INTO program(id,df_id,name,path) VALUES (vprogram_id,vdf_id,vname,vpath);
	END IF;
	INSERT INTO use_program(dt_id,program_id) VALUES (vdt_id,vprogram_id);
	RETURN vprogram_id;
END;

-- DROP FUNCTION insertDataSet;
CREATE FUNCTION insertDataSet (vdf_id INTEGER, vdt_id INTEGER, vdep_dt_id INTEGER, vtag VARCHAR(500), vtype VARCHAR(10))
RETURNS INTEGER
BEGIN
	DECLARE vds_id INTEGER;
    SELECT id INTO vds_id FROM data_set ds WHERE df_id = vdf_id AND tag = vtag;

    IF(vds_id IS NULL) THEN
    	#SELECT NEXT VALUE FOR "ds_id_seq" into vds_id;
    	INSERT INTO data_set(df_id,tag) VALUES (vdf_id,vtag);
    	SELECT id INTO vds_id FROM data_set ds WHERE df_id = vdf_id AND tag = vtag;
	END IF;

	IF(vdep_dt_id IS NOT NULL) THEN
		DECLARE vdd_id INTEGER;
		SELECT ds_id INTO vdd_id FROM data_dependency
		WHERE previous_dt_id = vdep_dt_id AND next_dt_id = vdt_id AND ds_id = vds_id;

		DECLARE vid INTEGER;
		SELECT id INTO vid FROM data_dependency WHERE previous_dt_id = vdep_dt_id AND ds_id = vds_id;

		IF(vid IS NULL) THEN
			IF(vdd_id IS NULL) THEN
				DECLARE vdd_id INTEGER;
				#SELECT NEXT VALUE FOR "dd_id_seq" into vdd_id;
				INSERT INTO data_dependency(previous_dt_id,next_dt_id,ds_id) VALUES (vdep_dt_id,vdt_id,vds_id);
			END IF;
		ELSE
			UPDATE data_dependency SET next_dt_id = vdt_id WHERE id = vid AND ds_id = vds_id;
		END IF;
	ELSE
		DECLARE vdd_id INTEGER;
		#SELECT NEXT VALUE FOR "dd_id_seq" into vdd_id;

		IF(vtype = 'INPUT') THEN
			INSERT INTO data_dependency(previous_dt_id,next_dt_id,ds_id) VALUES (null,vdt_id,vds_id);
		ELSE
			INSERT INTO data_dependency(previous_dt_id,next_dt_id,ds_id) VALUES (vdt_id,null,vds_id);
		END IF;
	END IF;

	RETURN vds_id;
END;

-- DROP FUNCTION insertAttribute;
CREATE FUNCTION insertAttribute (dds_id INTEGER, vextractor_id INTEGER, vname VARCHAR(30), vtype VARCHAR(15))
RETURNS INTEGER
BEGIN
	DECLARE vid INTEGER;
    SELECT id INTO vid FROM attribute WHERE ds_id=dds_id AND name=vname;
    IF(vid IS NULL) THEN
    	#SELECT NEXT VALUE FOR "att_id_seq" into vid;
    	INSERT INTO attribute(ds_id,extractor_id,name,type) VALUES (dds_id,vextractor_id,vname,vtype);
	END IF;
	RETURN SELECT id FROM attribute WHERE ds_id=dds_id AND name=vname;
END;

CREATE FUNCTION insertScientist (v_email VARCHAR(50))
RETURNS INTEGER
BEGIN
	DECLARE v_scientist_id INTEGER;
    SELECT sc.id INTO v_scientist_id FROM data_scientist sc WHERE sc.email=v_email;
    IF(v_scientist_id IS NULL) THEN
    	#SELECT NEXT VALUE FOR "df_id_seq" into v_df_id;
    	INSERT INTO data_scientist(email) VALUES (v_email);
	END IF;
	RETURN SELECT sc.id FROM data_scientist sc WHERE sc.email=v_email;
END;


CREATE FUNCTION insertHardwareInfo(
    vhostname VARCHAR(50), 
    vos VARCHAR(50), 
    vplatform VARCHAR(50), 
    varchitecture VARCHAR(50), 
    vprocessor VARCHAR(50), 
    vramTotal DOUBLE, 
    vramLimit DOUBLE, 
    vdiskTotal DOUBLE, 
    vdiskUsed DOUBLE, 
    vdiskFree DOUBLE, 
    vgpus VARCHAR(50), 
    vinContainer BOOLEAN
)
RETURNS INTEGER
BEGIN
    DECLARE v_hw_id INTEGER;

    SELECT id INTO v_hw_id FROM hardware_info
    WHERE hostname = vhostname
      AND os = vos
      AND platform = vplatform
      AND architecture = varchitecture
      AND processor = vprocessor
      AND ram_total_gb = vramTotal
      AND ram_limit_gb = vramLimit
      AND disk_total_gb = vdiskTotal
      AND disk_used_gb = vdiskUsed
      AND disk_free_gb = vdiskFree
      AND gpus = vgpus
      AND in_container = vinContainer;

    IF (v_hw_id IS NULL) THEN
        INSERT INTO hardware_info (
            hostname, os, platform, architecture, processor,
            ram_total_gb, ram_limit_gb,
            disk_total_gb, disk_used_gb, disk_free_gb,
            gpus, in_container
        ) VALUES (
            vhostname, vos, vplatform, varchitecture, vprocessor,
            vramTotal, vramLimit,
            vdiskTotal, vdiskUsed, vdiskFree,
            vgpus, vinContainer
        );
	END IF;
	RETURN SELECT id FROM hardware_info
		WHERE hostname IS NOT DISTINCT FROM vhostname
		AND os IS NOT DISTINCT FROM vos
		AND platform IS NOT DISTINCT FROM vplatform
		AND architecture IS NOT DISTINCT FROM varchitecture
		AND processor IS NOT DISTINCT FROM vprocessor
		AND ram_total_gb IS NOT DISTINCT FROM vramTotal
		AND ram_limit_gb IS NOT DISTINCT FROM vramLimit
		AND disk_total_gb IS NOT DISTINCT FROM vdiskTotal
		AND disk_used_gb IS NOT DISTINCT FROM vdiskUsed
		AND disk_free_gb IS NOT DISTINCT FROM vdiskFree
		AND gpus IS NOT DISTINCT FROM vgpus
		AND in_container IS NOT DISTINCT FROM vinContainer;

END;


-- DROP FUNCTION insertDataflowExecution;
CREATE FUNCTION insertDataflowExecution (etag VARCHAR(50),edf_id INTEGER, esc_id INTEGER, ehw_id INTEGER)
RETURNS INTEGER
BEGIN
	DECLARE id INTEGER;
    INSERT INTO dataflow_execution(tag,df_id,scientist_id,hardware_id) VALUES (etag,edf_id,esc_id,ehw_id);
	RETURN SELECT dfe.id FROM dataflow_execution dfe WHERE dfe.id=get_value_for('public', 'exec_id_seq')-1;
END;

-- DROP FUNCTION insertTask;
CREATE FUNCTION insertTask (videntifier INTEGER, vdf_tag VARCHAR(50), vdt_tag VARCHAR(50), vstatus VARCHAR(10), vworkspace VARCHAR(500), 
	vcomputing_resource VARCHAR(100), voutput_msg TEXT, verror_msg TEXT, vdf_exec VARCHAR(50))
RETURNS INTEGER
BEGIN
	DECLARE vid INTEGER;
	DECLARE vvstatus VARCHAR(10);
	DECLARE vdf_version INTEGER;
	DECLARE vdt_id INTEGER;
	DECLARE ve_id INTEGER;

	SELECT dfv.version, dt.id, dfe.id INTO vdf_version, vdt_id, ve_id
	FROM dataflow df, data_transformation dt, dataflow_execution dfe, dataflow_version as dfv
	WHERE df.id = dt.df_id AND dfv.df_id = df.id AND dfe.df_id = df.id AND df.tag = vdf_tag AND dt.tag = vdt_tag AND dfe.tag = vdf_exec;

	IF((vdf_version IS NOT NULL) AND (vdt_id IS NOT NULL) AND (ve_id IS NOT NULL)) THEN
		SELECT t.id, t.status INTO vid, vvstatus
		FROM task t
		WHERE t.df_version = vdf_version AND t.dt_id = vdt_id AND t.identifier = videntifier AND t.df_exec = vdf_exec;

		IF(vid IS NULL) THEN
		    #SELECT NEXT VALUE FOR "task_id_seq" into vid;
		    INSERT INTO task(identifier,df_version,df_exec,dt_id,status,workspace,computing_resource,output_msg,error_msg) 
		    VALUES (videntifier,vdf_version,vdf_exec,vdt_id,vstatus,vworkspace,vcomputing_resource,voutput_msg,verror_msg);
	    ELSE
	    	UPDATE task
	    	SET status = vstatus, output_msg = voutput_msg, error_msg = verror_msg
	    	WHERE identifier = videntifier AND df_version = vdf_version AND dt_id = vdt_id AND df_exec = vdf_exec;
		END IF;
    END IF;
	RETURN SELECT t.id FROM task t WHERE t.df_version = vdf_version AND t.dt_id = vdt_id AND t.identifier = videntifier AND t.df_exec = vdf_exec;
END;

-- DROP FUNCTION insertFile;
CREATE FUNCTION insertFile (vtask_id INTEGER, vname VARCHAR(200), vpath VARCHAR(500))
RETURNS INTEGER
BEGIN
	DECLARE vid INTEGER;
    SELECT id INTO vid FROM file WHERE name=vname AND path=vpath;
    IF(vid IS NULL) THEN
    	SELECT NEXT VALUE FOR "public"."file_id_seq" into vid;
    	INSERT INTO file(id,task_id,name,path) VALUES (vid,vtask_id,vname,vpath);
	END IF;
	RETURN vid;
END;

-- DROP FUNCTION insertPerformance;
CREATE FUNCTION insertPerformance (vtask_id INTEGER, vsubtask_id INTEGER, vmethod VARCHAR(30), vdescription VARCHAR(200), vstarttime VARCHAR(30), vendtime VARCHAR(30), vinvocation TEXT)
RETURNS INTEGER
BEGIN
	DECLARE vid INTEGER;
	IF(vsubtask_id IS NULL) THEN
		SELECT id INTO vid FROM performance WHERE method=vmethod and task_id=vtask_id;
	ELSE
		SELECT id INTO vid FROM performance WHERE method=vmethod and task_id=vtask_id and subtask_id=vsubtask_id;
	END IF;
    
    IF(vid IS NULL) THEN
    	#SELECT NEXT VALUE FOR "performance_id_seq" into vid;
    	INSERT INTO performance(task_id,subtask_id,method,description,starttime,endtime,invocation) VALUES (vtask_id,vsubtask_id,vmethod,vdescription,vstarttime,vendtime,vinvocation);
	ELSE
    	UPDATE performance
    	SET endtime = vendtime, invocation = vinvocation
    	WHERE id = vid and endtime = 'null';
	END IF;
	
	IF(vsubtask_id IS NULL) THEN
		RETURN SELECT id FROM performance WHERE method=vmethod and task_id=vtask_id;
	ELSE
		RETURN SELECT id FROM performance WHERE method=vmethod and task_id=vtask_id and subtask_id=vsubtask_id;
	END IF;
END;


-- DROP FUNCTION insertExtractor;
CREATE FUNCTION insertExtractor (vtag VARCHAR(20), vds_id INTEGER, vcartridge VARCHAR(20), vextension VARCHAR(20))
RETURNS INTEGER
BEGIN
	DECLARE vid INTEGER;
    SELECT id INTO vid FROM extractor WHERE tag = vtag AND ds_id = vds_id AND cartridge = vcartridge AND extension = vextension;
    IF(vid IS NULL) THEN
    	SELECT NEXT VALUE FOR "public"."extractor_id_seq" into vid;
    	INSERT INTO extractor(id,ds_id,tag,cartridge,extension) VALUES (vid,vds_id,vtag,vcartridge,vextension);
	END IF;
	RETURN vid;
END;

-- DROP FUNCTION insertExtractorCombination;
CREATE FUNCTION insertExtractorCombination (vouter_ext_id INTEGER, vinner_ext_id INTEGER, vds_id INTEGER, vkeys VARCHAR(100), vkey_types VARCHAR(100))
RETURNS INTEGER
BEGIN
	DECLARE vid INTEGER;
    SELECT id INTO vid FROM extractor_combination WHERE outer_ext_id = vouter_ext_id AND inner_ext_id = vinner_ext_id AND ds_id = vds_id;
    IF(vid IS NULL) THEN
    	SELECT NEXT VALUE FOR "public"."ecombination_id_seq" into vid;
    	INSERT INTO extractor_combination(outer_ext_id,inner_ext_id,keys,key_types,ds_id) VALUES (vouter_ext_id,vinner_ext_id,vkeys,vkey_types,vds_id);
	END IF;
	RETURN vid;
END;

COMMIT;

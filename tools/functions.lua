function copydir(src_dir, dst_dir, filter, single_dst_dir)
    filter = filter or "**"
    src_dir = src_dir .. "/"
    print('copy "' .. src_dir .. filter .. '" to "' .. dst_dir .. '".')
    dst_dir = dst_dir .. "/"
    local dir = path.rebase(".", path.getabsolute("."), src_dir) -- root dir, relative from src_dir

    os.chdir(src_dir) -- change current directory to src_dir
    local matches = os.matchfiles(filter)
    os.chdir(dir) -- change current directory back to root

    local counter = 0
    for k, v in ipairs(matches) do
        local target = iif(single_dst_dir, path.getname(v), v)
        --make sure, that directory exists or os.copyfile() fails
        os.mkdir(path.getdirectory(dst_dir .. target))
        if os.copyfile(src_dir .. v, dst_dir .. target) then
            counter = counter + 1
        end
    end

    if counter == #matches then
        print(counter .. " files copied.")
        return true
    else
        print("Error: " .. counter .. "/" .. #matches .. " files copied.")
        return nil
    end
end

function file_exists(file)
    local f = io.open(file, "rb")
    if f then f:close() end
    return f ~= nil
end
  
function lines_from(file)
    if not file_exists(file) then return {} end
    local lines = {}
    for line in io.lines(file) do 
      lines[#lines + 1] = line
    end
    return lines
end

function read_file(file)
    local lines = lines_from(file)
    str = ''
    for num, line in pairs(lines) do
        str = str..line.."\n"
    end
    return str
end

function get_version(file)
    local lines = lines_from(file)
    major = tonumber(lines[1])
    minor = tonumber(lines[2])
    patch = "0x"..(lines[3])
    return major, minor, patch
end

-- find the path of 'Hipcc' from PATH
-- return nil if not exist
-- only works for linux ( for now )
function find_hipcc_path()

	if os.host() ~= "linux" then
		return nil
	end

	local cmd = 'which hipcc 2>/dev/null'

	local f = io.popen(cmd)
	local hipcc_path = f:read("*a")
	f:close()

	if hipcc_path == nil or hipcc_path == '' then
		print("hipcc_path nil");
		return nil
	else
		print("-- hipcc_path = " .. hipcc_path );
		-- Remove any trailing whitespace
		hipcc_path = hipcc_path:gsub("%s+$", "")

		-- Extract the directory from the full path
		local dir = hipcc_path:match("(.+)/[^/]+$")
		return dir
	end
end

function get_hip_sdk_verion()
	
	if os.ishost("windows") then
		root = '.\\'
	end
	
	hip_command = 'hipcc'
	HIP_PATH = os.getenv("HIP_PATH")
	PATH = os.getenv("PATH")
	
	hipccFromPATH = find_hipcc_path()
	if fromPATH ~= nil then
		print( "hipcc found from PATH: ".. hipccFromPATH )
	end
	
	if os.ishost("windows") then
		if not HIP_PATH then
			-- if the HIP_PATH env var is not set, we assume there is a 'hipSdk' folder at the root of the project.
			HIP_PATH = path.getabsolute(root .. 'hipSdk') -- convert the path to absolute
		end
	
		if string.sub(HIP_PATH, -1, -1) == '\\' or string.sub(HIP_PATH, -1, -1) == '/' then
			HIP_PATH = string.sub(HIP_PATH, 1, -2)
		end
		
		-- HIP_PATH is expected to look like:   C:\Program Files\AMD\ROCm\5.7
		print("using HIP_PATH = " .. HIP_PATH)
		
		if os.isfile(HIP_PATH .. '\\bin\\hipcc.exe') then
			-- in newer version of HIP SDK (>= 6.3), we are using 'hipcc.exe --version' to check the version
			-- print("using hipcc.exe to get the version.")
			hip_command = '\"' .. HIP_PATH..'\\bin\\hipcc.exe\"'
		elseif os.isfile(HIP_PATH .. '\\bin\\hipcc') then
			-- in older version of HIP SDK, we are using 'perl hipcc --version' to check the version
			-- print("using perl hipcc to get the version.")
			hip_command = '\"' .. HIP_PATH..'\\bin\\hipcc\"'
		else
			print("ERROR: hipcc.exe or hipcc not found in the SDK path.")
			hip_command = 'hipcc'
		end
	
	-- for LINUX
	else
		if not HIP_PATH then
			if hipccFromPATH ~= nil then
				hip_command = 'hipcc'
			end
			
		-- if HIP_PATH is set, we take the path from it.
		else
			if string.sub(HIP_PATH, -1, -1) == '\\' or string.sub(HIP_PATH, -1, -1) == '/' then
				HIP_PATH = string.sub(HIP_PATH, 1, -2)
			end
			
			hip_command = '\"' .. HIP_PATH..'/bin/hipcc\"'
		end
	end
	
	tmp_file = os.tmpname ()
	fullcommand = hip_command .. " --version > " .. tmp_file
	print("Executing: " .. fullcommand);
	os.execute (fullcommand)
	
	local version
	for line in io.lines (tmp_file) do
		print (line)
		version =  string.sub(line, string.find(line, "%d.%d"))
		break
	end
	os.remove (tmp_file)

    if version == nil or version == '' then
        version = "HIP_SDK_NOT_FOUND"
    end

	return version, HIP_PATH
end

function write_version_info(in_file, header_file, version_file, hip_sdk_version)
	if not file_exists(version_file) then
		print("Version.txt file missing!\n")
		return
	end
	if not file_exists(in_file) then 
		print(string.format("%s file is missing!\n", in_file))
		return
	end
	
	HIPRT_MAJOR_VERSION, HIPRT_MINOR_VERSION, HIPRT_PATCH_VERSION = get_version(version_file)
	HIPRT_VERSION = HIPRT_MAJOR_VERSION * 1000 + HIPRT_MINOR_VERSION 
	HIPRT_API_VERSION = HIPRT_VERSION 
	HIPRT_VERSION_STR = string.format("%05d", HIPRT_VERSION)
	print( "HIPRT_API_VERSION: "..HIPRT_VERSION_STR .."_".. HIPRT_PATCH_VERSION )
	header = read_file(in_file)
	header = header:gsub("@HIPRT_MAJOR_VERSION@", HIPRT_MAJOR_VERSION)
	header = header:gsub("@HIPRT_MINOR_VERSION@", HIPRT_MINOR_VERSION)
	header = header:gsub("@HIPRT_PATCH_VERSION@", HIPRT_PATCH_VERSION)
	header = header:gsub("@HIPRT_API_VERSION@", HIPRT_API_VERSION)
	header = header:gsub("@HIPRT_VERSION_STR@", "\""..HIPRT_VERSION_STR.."\"")
	header = header:gsub("@HIP_VERSION_STR@", "\""..hip_sdk_version.."\"")
	file = io.open(header_file, "w")
	file:write(header)
	file:close()
end

function get_hiprt_library_name(version_file)
	HIPRT_MAJOR_VERSION, HIPRT_MINOR_VERSION, HIPRT_PATCH_VERSION = get_version(version_file)
	HIPRT_VERSION = HIPRT_MAJOR_VERSION * 1000 + HIPRT_MINOR_VERSION 
	HIPRT_VERSION_STR = string.format("%05d", HIPRT_VERSION)
	return  "hiprt" .. HIPRT_VERSION_STR
end

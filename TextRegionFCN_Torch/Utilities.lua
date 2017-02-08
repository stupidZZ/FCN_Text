function setupLogger(fpath)
    local fileMode = 'w'
    if paths.filep(fpath) then
        local input = nil
        while not input do
            print('Logging file exits, overwrite(o)? append(a)? abort(q)?')
            input = io.read()
            if input == 'o' then
                fileMode = 'w'
            elseif input == 'a' then
                fileMode = 'a'
            elseif input == 'q' then
                os.exit()
            else
                fileMode = nil
            end
        end
    end
    gLoggerFile = io.open(fpath, fileMode)
end

function shutdownLogger()
    if gLoggerFile then
        gLoggerFile:close()
    end
end

function logging(message)
    local M
    if type(message) == 'table' then
        M = message
    else
        M = {message}
    end

    for i = 1,#M do
        local timeStamp = os.date('%x %X')
        local msgFormatted = string.format('[%s]  %s', timeStamp, M[i])
        print(msgFormatted)
        if gLoggerFile then
            gLoggerFile:write(msgFormatted .. '\n')
            gLoggerFile:flush()
        end
    end
end

function file_exists(path)
    local file = io.open(path, "rb")
    if file then 
        file:close() 
    end
    return file ~= nil
end
